import numpy as np
from scipy.linalg import block_diag,sqrtm
import torch
from data_gen import batch_data_gen_quantized,batch_data_gen_unquantized,uniform_quantizer,get_data_symbol,generate_covariance_matrices
from parameters import parameter_reading
import matplotlib.pyplot as plt
from train import complex_to_vec,LS_ICL_token_processing,ICL_token_processing
import pickle

def ICL_equalizer(model,args,Rs_te):
    '''Testing In-Context Equalizer
    Model: Model to test
    args: Simulation parameters for pre-processing (see parameters.py)
    Rs_te: Covariance Matrices to test

    Return: mean MSE, std MSE, mean SER, std SER
    '''
    R_flat_te = [np.stack([block_diag(*[r[:, :, k, i] for k in range(0, r.shape[2])]) for i in range(0, r.shape[3])], axis=-1) for r in Rs_te]
    ls_coeff_te = [np.stack([np.diag(r[:, :, i]) for i in range(0, r.shape[2])], axis=-1) for r in R_flat_te]
    L_flat_te = [np.stack([np.linalg.cholesky(r[:, :, i]) for i in range(0, r.shape[2])], axis=-1) for r in R_flat_te]
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        '''Forward Pass'''
        X, Y, M, R = batch_data_gen_quantized(len(Rs_te), args=args, seed_example=2 ** 32 - 1,cov_mats=Rs_te,L_mats=L_flat_te,ls_coeffs=ls_coeff_te)
        if args.large_scale_ICL:
            te_x, te_y = LS_ICL_token_processing(X, Y, M, R, args)
        else:
            te_x, te_y = ICL_token_processing(X, Y, M, args)
        te_x = torch.Tensor(np.stack(te_x, axis=0)).to(device)
        te_y = torch.Tensor(np.stack(te_y, axis=0)).to(device)
        output = model(te_y, te_x)
        x_hat = output[:, -1, 0::2] + 1j * output[:, -1, 1::2]
        x_hat = x_hat.detach().cpu().numpy()
        te_x = te_x.detach().cpu().numpy()
        '''MSE Computation'''
        te_x = np.stack([te_x[:, :, i * 2:(i + 1) * 2] for i in range(0, int(te_x.shape[-1] / 2))])
        te_x = np.reshape(np.swapaxes(te_x, 0, 1), output.shape)
        te_x=te_x[:,-1,0::2]+1j*te_x[:,-1,1::2]
        loss_mean = np.mean(np.real(x_hat - te_x) ** 2 + np.imag(x_hat - te_x) ** 2)
        loss_std = np.std(np.real(x_hat - te_x) ** 2 + np.imag(x_hat - te_x) ** 2)
        '''SER Computation'''
        x_hat=np.reshape(x_hat,(-1,args.numofUE))
        te_x = np.reshape(te_x, (-1, args.numofUE))
        mods_map = [4, 16, 64, 2, 8]
        mods_te_id = [mods_map[int(m[0] * 4.)] for m in M]
        const_te = [get_data_symbol(args, symbols=np.arange(0, n_c))[0] for n_c in mods_te_id]
        SER_mean = 0
        SER_std = 0
        for nue in range(args.numofUE):
            s_hat = np.stack([np.argmin(np.abs(const_te[i] - x_hat[i, nue]), axis=0) for i in range(len(const_te))])
            s = np.stack([np.argmin(np.abs(const_te[i] - te_x[i, nue]), axis=0) for i in range(len(const_te))])
            SER_mean = SER_mean + np.mean(np.abs(s - s_hat) > 0.1)
            SER_std = SER_std + np.mean(np.abs(s - s_hat) > 0.1)
    return loss_mean, loss_std, SER_mean/args.numofUE, SER_std/args.numofUE

def centralized_LMMSE(args,Rs_te,UNQUANTIZED=False):
    '''
    Testing Centralized MMSE

    args: Simulation parameters for pre-processing (see parameters.py)
    Rs_te: Covariance Matrices to test
    UNQUANTIZED: Set to True if want to remove quantization

    Return: mean MSE, std MSE, mean SER, std SER
    '''

    R_flat_te = [np.stack([block_diag(*[r[:, :, k, i] for k in range(0, r.shape[2])]) for i in range(0, r.shape[3])], axis=-1) for r in Rs_te]
    ls_coeff_te = [np.stack([np.diag(r[:, :, i]) for i in range(0, r.shape[2])], axis=-1) for r in R_flat_te]
    L_flat_te = [np.stack([np.linalg.cholesky(r[:, :, i].astype(np.complex64)) for i in range(0, r.shape[2])], axis=-1) for r in R_flat_te]
    n_te=len(Rs_te)
    noiseVar = 10 ** (args.noise_power_dB / 10.)
    if UNQUANTIZED:
        te_x, te_y, mods_te = batch_data_gen_unquantized(n_te, args=args, seed_example=2 ** 32 - 1, cov_mats=Rs_te,L_mats=L_flat_te, TESTING=True)
        bussgang_gain=np.eye(args.numofAnt*args.numofAP)
        distortion=0
    else:
        te_x, te_y, mods_te,_ = batch_data_gen_quantized(n_te, args=args, seed_example=2 ** 32 - 1, cov_mats=Rs_te,L_mats=L_flat_te,ls_coeffs=ls_coeff_te, TESTING=True,SCALING=False)
        '''Get the unquantized data to compute the Bussgang Decomposition'''
        te_x_u, te_y_u, mods_te_u = batch_data_gen_unquantized(n_te, args=args, seed_example=2 ** 32 - 1, cov_mats=Rs_te,L_mats=L_flat_te, TESTING=True)
        bussgang_gain=np.mean(np.multiply(np.stack(te_y),(np.stack(te_y_u).conj())))/np.mean(np.abs(np.stack(te_y_u))**2)*np.eye(args.numofAnt*args.numofAP)
        distortion = np.mean(np.abs(np.stack(te_y_u)@bussgang_gain-np.stack(te_y))**2)
    mods_map=[4,16,64,2,8]
    mods_te_id = [mods_map[int(m[0] * 4.)] for m in mods_te]
    const_te=[get_data_symbol(args, symbols=np.arange(0, n_c))[0] for n_c in mods_te_id]
    te_x = np.stack(te_x)
    te_y = np.stack(te_y)
    Rs_te=np.stack(Rs_te,axis=-1)
    '''Extracting Tx Pilots and Data, Reshaping to [batch size, number of EU, length]'''
    pilot = np.transpose(te_x[:, :args.dimPilotMatrix, :], (0, 2, 1))
    data = np.transpose(te_x[:, args.dimPilotMatrix:, :], (0, 2, 1))
    '''Extracting Rx Pilots and Data, Reshaping to [batch size, number of EU, length]'''
    y_pilot = np.transpose(te_y[:, :args.dimPilotMatrix, :], (0, 2, 1))
    y_data = np.transpose(te_y[:, args.dimPilotMatrix:, :], (0, 2, 1))
    '''Correlate Received Signal with Pilot'''
    h_hats=[]
    Cs=[]
    R_u=[]
    '''Obtain Block diagonal Channel Matrix from each UE to APs'''
    for k in range(0, args.numofUE):
        Rs_u = Rs_te[:, :, :, k, :]
        R_u.append([block_diag(*[Rs_u[:, :, k, i] for k in range(0, args.numofAP)]) for i in range(0, Rs_u.shape[-1])])
    '''Channel Estimation for Each UE'''
    for k in range(0,args.numofUE):
        pilot_u=pilot[:,k,:]  #Pilot of UE k
        '''Computing set of UEs that assigned to pilot_u'''
        diff=np.sum(np.abs(pilot_u[:,np.newaxis,:]-pilot),axis=-1)
        collision_id= [np.where(diff[i,:]==0)[0]  for i in range(diff.shape[0])]
        z_u= [y_pilot[i, :, :] @ (pilot_u[i,:].conj())/ np.sqrt(args.dimPilotMatrix) for i in range(0, y_pilot.shape[0])]
        Rs_u_sum=[ np.sum([R_u[j][i] for j in collision_id[i]],axis=0) for i in range(n_te)]
        Rs_u_tot=[ np.sum([R_u[j][i] for j in range(0,args.numofUE)],axis=0) for i in range(n_te)]
        Psi_inv=[np.linalg.inv(((bussgang_gain@Rs_u_sum[i]@(bussgang_gain.conj()))*args.dimPilotMatrix+bussgang_gain@(noiseVar*np.eye(args.numofAnt*args.numofAP))@(bussgang_gain.conj()))+distortion*(noiseVar*np.eye(args.numofAnt*args.numofAP)+Rs_u_tot[i])) for i in range(n_te)]
        h_hat=np.asarray([np.sqrt(args.dimPilotMatrix)*R_u[k][i]@bussgang_gain.conj()@Psi_inv[i]@z_u[i] for i in range(n_te)])
        h_hats.append(h_hat)
        C=np.asarray([R_u[k][i]-args.dimPilotMatrix*R_u[k][i]@(bussgang_gain.conj())@Psi_inv[i]@bussgang_gain@R_u[k][i] for i in range(n_te)])
        Cs.append(C)
    sum_inner_hs=[np.sum([np.expand_dims(h_hats[k][i,:],axis=-1)@np.expand_dims(h_hats[k][i,:],axis=-1).conj().T for k in range(0, args.numofUE)],axis=0) for i in range(0,n_te)]
    sum_Cs=np.sum(Cs,axis=0)
    X_hat=[]
    S_hat=[]
    S=[]
    '''Computation of the combining vector and linear equalization'''
    for k in range(0, args.numofUE):
        combiner=[np.expand_dims(np.linalg.inv(sum_inner_hs[i]+sum_Cs[i]+noiseVar*np.eye(args.numofAnt*args.numofAP))@h_hats[k][i,:],axis=-1) for i in range(0,n_te)]
        x_hat=[ combiner[i].conj().T@y_data[i] for i in range(0,n_te)]
        x_hat=np.squeeze(np.asarray(x_hat))
        X_hat.append(x_hat)
        s_hat = np.stack([ np.argmin(np.abs(const_te[i] - x_hat[i]), axis=0) for i in range(len(const_te))])
        s = np.stack([np.argmin(np.abs(const_te[i] - data[i,k,:]), axis=0) for i in range(len(const_te))])
        S_hat.append(s_hat)
        S.append(s)
    X_hat=np.stack(X_hat,axis=-1)
    '''Mean and standard deviation of the MSE'''
    loss_mean=np.mean(np.real(X_hat - data[:,:,0]) ** 2 + np.imag(X_hat - data[:,:,0]) ** 2)
    loss_std=np.std(np.real(X_hat - data[:,:,0]) ** 2 + np.imag(X_hat - data[:,:,0]) ** 2)
    S=np.asarray(S)
    S_hat=np.asarray(S_hat)
    '''Mean and standard deviation of the SER'''
    SER_mean=np.mean(np.abs(S - S_hat) > 0.1)
    SER_std = np.std(np.abs(S - S_hat) > 0.1)
    return loss_mean, loss_std, SER_mean, SER_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found Device: '+str(device))
args = parameter_reading()

''' This first part is to test the pilot Reuse'''
args.large_scale_ICL=True
if args.large_scale_ICL:
    args.model_directory = './LS_ICL_saved_model_n_layers_' + str(args.num_layer)
else:
    args.model_directory = './ICL_saved_model_n_layers_' + str(args.num_layer)
colors=['tab:red','tab:blue','tab:green','tab:grey']
markers=['o','X','s','^']
lss=['-.','--',':','-']
args.n_tasks=8192
i = 0
seed_setup = 3
N_UE=[4]
n_te_setup=1000
Pow_N_TE=-np.arange(10,70,5)
RS_TEs=[generate_covariance_matrices(args.numofAP, n_ue,n_ue, args.numofAnt, n_te_setup, seed_setup + 2) for n_ue in N_UE]
BITS_TE=[8]
args.modulationDiversity = True
args.modulationAware = True
args.numberUEAware = False
Centr_u = []
args.pilot_contamination=False
for args.n_reuse in [0,1,2,3]:
    Centr_temp_u = []
    for p_noise in Pow_N_TE:
        args.noise_power_dB = p_noise
        for Rs_te, n_ue in zip(RS_TEs, N_UE):
            args.numofUE = n_ue
            Centr_temp_u.append(centralized_LMMSE(args, Rs_te,UNQUANTIZED=True))
    Centr_u.append(Centr_temp_u)
Centr_u = np.asarray(Centr_u)

ICL_Div, Centr = [], []
for args.n_reuse in [0,1,2,3]:
    b=BITS_TE[0]
    args.bits = b
    args.numofUE_min = 1
    args.numofUE_max = 4
    args.quantization_levels = np.load("lloyd_codebooks.npz", allow_pickle=True)['arr_0'][int(args.bits - 1)]
    ICL_Div_temp, Centr_temp = [], []
    for p_noise in Pow_N_TE:
        ICL_noise_pow = -50
        print('Noise: ' + str(p_noise))
        args.noise_power_dB = p_noise
        for Rs_te, n_ue in zip(RS_TEs, N_UE):
            args.numofUE = n_ue
            model_path = args.model_directory + '/modAware_' + str(args.modulationAware) + '_nUEAware_' + str(args.numberUEAware) + '_numofAP_' + str(args.numofAP) + '_numofUEmin_' + str(args.numofUE_min) + '_numofUEmax_' + str(args.numofUE_max) + '_modulationScheme_Diversity_tasks_' + str(
                args.n_tasks) + '_NoisePowerdB_' + str(ICL_noise_pow) + '_bits_' + str(b) + '.pth'
            model = torch.load(model_path, map_location=torch.device(device))
            ICL_Div_temp.append(ICL_equalizer(model, args, Rs_te))
            Centr_temp.append(centralized_LMMSE(args, Rs_te))
    Centr.append(Centr_temp)
    ICL_Div.append(ICL_Div_temp)
Centr = np.asarray(Centr)
ICL_Div = np.asarray(ICL_Div)
dictionary = {
  "POWS": Pow_N_TE,
  "ICL":ICL_Div,
  "Centr":  Centr,
  "Centr_u": Centr_u,
}
if args.large_scale_ICL and args.modulationAware:
    with open('Results/PilotReuseLSBits_'+str(b)+'.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
elif args.large_scale_ICL and not args.modulationAware:
    with open('Results/PilotReuseLS_ModUnawareBits_'+str(b)+'.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
elif not args.large_scale_ICL and args.modulationAware:
    with open('Results/PilotReuseBits_'+str(b)+'.pkl', 'wb') as f:
        pickle.dump(dictionary, f)


''' This second part is to test the quantization'''
i = 0
seed_setup = 3
N_UE=[1,2,3,4]
n_te_setup=1000
Pow_N_TE=-np.arange(10,70,5)
RS_TEs=[generate_covariance_matrices(args.numofAP, n_ue,n_ue, args.numofAnt, n_te_setup, seed_setup + 2) for n_ue in N_UE]
BITS_TE=[1,2,3,4,5,6,7,8]
args.modulationDiversity = True
args.modulationAware = True
args.numberUEAware = False
args.n_reuse=0
for args.pilot_contamination in [False,True]:
    p_noise=-50
    Centr_u = []
    for b in BITS_TE:
        args.noise_power_dB = p_noise
        Centr_temp_u, Local_temp_u = [], []
        for Rs_te, n_ue in zip(RS_TEs, N_UE):
            args.numofUE = n_ue
            Centr_temp_u.append(centralized_LMMSE(args, Rs_te,UNQUANTIZED=True))
        Centr_u.append(np.mean(Centr_temp_u, axis=0))
    Centr_u = np.asarray(Centr_u)

    ICL_Div, Centr = [], []
    for b in BITS_TE:
        ICL_noise_pow = -50
        print('Bits: ' + str(b))
        args.noise_power_dB = p_noise
        args.bits = b
        args.quantization_levels = np.load("lloyd_codebooks.npz", allow_pickle=True)['arr_0'][int(args.bits - 1)]
        args.numofUE_min = 1
        args.numofUE_max = 4
        ICL_Div_temp, Centr_temp = [], []
        for Rs_te, n_ue in zip(RS_TEs, N_UE):
            args.numofUE = n_ue
            model_path = args.model_directory + '/modAware_' + str(args.modulationAware) + '_nUEAware_' + str(args.numberUEAware) + '_numofAP_' + str(args.numofAP) + '_numofUEmin_' + str(args.numofUE_min) + '_numofUEmax_' + str(args.numofUE_max) + '_modulationScheme_Diversity_tasks_' + str(
                args.n_tasks) + '_NoisePowerdB_' + str(ICL_noise_pow) + '_bits_' + str(b) + '.pth'
            model = torch.load(model_path, map_location=torch.device(device))
            ICL_Div_temp.append(ICL_equalizer(model, args, Rs_te))
            Centr_temp.append(centralized_LMMSE(args, Rs_te))
        Centr.append(np.mean(Centr_temp, axis=0))
        ICL_Div.append(np.mean(ICL_Div_temp, axis=0))
    Centr = np.asarray(Centr)
    ICL_Div = np.asarray(ICL_Div)
    dictionary = {
      "BITS": BITS_TE,
      "ICL":ICL_Div,
      "Centr":  Centr,
      "Centr_u": Centr_u,
    }
    if args.large_scale_ICL and args.modulationAware:
        if args.pilot_contamination:
            with open('Results/ResultsVsBitsContaminationLS.pkl', 'wb') as f:
                pickle.dump(dictionary, f)
        else:
            with open('Results/ResultsVsBitsNoContaminationLS.pkl', 'wb') as f:
                pickle.dump(dictionary, f)
    elif args.large_scale_ICL and not args.modulationAware:
        if args.pilot_contamination:
            with open('Results/ResultsVsBitsContaminationLS_ModUnaware.pkl', 'wb') as f:
                pickle.dump(dictionary, f)
        else:
            with open('Results/ResultsVsBitsNoContaminationLS_ModUnaware.pkl', 'wb') as f:
                pickle.dump(dictionary, f)
    elif not args.large_scale_ICL and not args.modulationAware:
        if args.pilot_contamination:
            with open('Results/ResultsVsBitsContamination_ModUnaware.pkl', 'wb') as f:
                pickle.dump(dictionary, f)
        else:
            with open('Results/ResultsVsBitsNoContamination_ModUnaware.pkl', 'wb') as f:
                pickle.dump(dictionary, f)
