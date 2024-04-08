import numpy as np
import torch
import torch.optim as optim
from data_gen import batch_data_gen_quantized, generate_covariance_matrices
import time
from scipy.linalg import block_diag
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(model, ys_batch, xs_batch, optimizer):
    model.train()
    ys_batch=torch.Tensor(np.stack(ys_batch,axis=0)).to(device)
    xs_batch=torch.Tensor(np.stack(xs_batch,axis=0)).to(device)
    output = model(ys_batch, xs_batch)
    optimizer.zero_grad()
    loss = mean_data_squared_error(output, xs_batch)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

def val_step(model, ys_batch, xs_batch):
    model.eval()
    ys_batch = torch.Tensor(np.stack(ys_batch, axis=0)).to(device)
    xs_batch = torch.Tensor(np.stack(xs_batch, axis=0)).to(device)
    output =model(ys_batch, xs_batch)
    norm_err = mean_data_squared_error(output, xs_batch)
    return norm_err.detach().item()

def mean_data_squared_error(xs_pred, xs):
    err = xs - xs_pred
    norm_err = torch.norm(err[:,-1:,:], dim=2)
    return norm_err.square().mean()

def complex_to_vec(X):
    "Converts complex matrix to real vector"
    real=np.real(X)
    imag=np.imag(X)
    X_vec = np.concatenate([np.real(X) , np.imag(X) ], axis=1)
    X_vec[:,0:-1:2]=real
    X_vec[:,1::2] = imag
    return X_vec

def LS_ICL_token_processing(X,Y,M,R,args):
    "Prompt processing using large scale fading information"
    R_flat = [np.stack([np.diag(block_diag(*[r[:, :, k, i] for k in range(0, r.shape[2])])) for i in range(0, r.shape[3])], axis=0) for r in R]
    pilots = [np.real(x[:-1, :]) for x in X]
    n_col = [[np.where(np.sum(np.abs(p - p[:, i:i + 1]), axis=0) == 0)[0].tolist() for i in range(p.shape[-1])] for p in pilots]
    [[n[i].remove(i) for i in range(len(n))] for n in n_col]
    [[n[i].insert(0, i) for i in range(len(n))] for n in n_col]
    add_token = [np.stack([np.real(np.pad(r[np.asarray(id), :], ((0, args.numofUE_max - len(id)), (0, 0)))) for id in n], axis=0) for n, r in zip(n_col, R_flat)]
    X, Y = [complex_to_vec(x) for x in X], [complex_to_vec(y) for y in Y]
    if args.modulationAware:
        Y = [np.concatenate((y, m), axis=1) for y, m in zip(Y, M)]
    tk_length = np.max([Y[0].shape[-1], add_token[0].shape[-1]])
    Y = [np.pad(y, ((0, 0), (0, tk_length - y.shape[-1]))) for y in Y]
    add_token = [np.pad(y, ((0, 0), (0, 0), (0, tk_length - y.shape[-1]))) for y in add_token]
    Y = np.concatenate([np.stack([y for _ in range(0, int(t.shape[0]))], axis=0) for y, t in zip(Y, add_token)], axis=0)
    x = np.concatenate([np.stack([x[:, i * 2:(i + 1) * 2] for i in range(0, int(x.shape[-1] / 2))], axis=0) for x in X], axis=0)
    add_token = np.concatenate(add_token, axis=0)
    y = np.concatenate((add_token, Y), axis=1)
    return x,y

def ICL_token_processing(X,Y,M,args):
    "Prompt processing without large scale fading information"
    X, Y = [complex_to_vec(x) for x in X], [complex_to_vec(y) for y in Y]
    if args.modulationAware:
        Y = [np.concatenate((y, m), axis=1) for y, m in zip(Y, M)]
    y = np.concatenate([np.stack([y for _ in range(0, int(x.shape[-1] / 2))], axis=0) for y, x in zip(Y, X)], axis=0)
    x = np.concatenate([np.stack([x[:, i * 2:(i + 1) * 2] for i in range(0, int(x.shape[-1] / 2))], axis=0) for x in X], axis=0)
    return x,y

def trainNetwork(model_GPT2,args):
    optimizer_model_GPT2 = optim.Adam(model_GPT2.parameters(), lr=args.learning_rate)
    seed_setup,seed_task=1,1
    '''Tasks for Validation'''
    Rs_val = generate_covariance_matrices(args.numofAP, args.numofUE_min, args.numofUE_max,args.numofAnt, args.val_data_size, seed_setup+1)
    '''Combining APs Spatial Correlation Matrices'''
    R_flat_val = [np.stack([block_diag(*[r[:, :, k, i] for k in range(0, r.shape[2])]) for i in range(0, r.shape[3])], axis=-1) for r in Rs_val]
    '''Large-scale Fading Coefficients'''
    ls_coeff_val = [np.stack([np.diag(r[:, :, i]) for i in range(0, r.shape[2])], axis=-1) for r in R_flat_val]
    '''Cholesky Factorization to generate circular Gaussians'''
    L_flat_val = [np.stack([np.linalg.cholesky(r[:, :, i]) for i in range(0, r.shape[2])], axis=-1) for r in R_flat_val]
    '''Generating Validation Data'''
    X, Y, M, R = batch_data_gen_quantized(args.val_data_size, args=args, seed_example=2**32-1,cov_mats=Rs_val,L_mats=L_flat_val,ls_coeffs=ls_coeff_val)
    '''Genration of training prompts'''
    if args.large_scale_ICL:
        x_val, y_val = LS_ICL_token_processing(X, Y, M, R, args)
    else:
        x_val, y_val = ICL_token_processing(X, Y, M, args)
    n_it_per_epoch=8
    N_ex=1024
    log_every,best_val  = 50,100
    nbrOfSetups_tr=args.n_tasks
    '''Tasks for Training followed by same pre-processing as Validation'''
    Rs_tr = generate_covariance_matrices(args.numofAP,  args.numofUE_min, args.numofUE_max, args.numofAnt, nbrOfSetups_tr, seed_setup)
    R_flat_tr = [np.stack([block_diag(*[r[:, :, k, i] for k in range(0, r.shape[2])]) for i in range(0, r.shape[3])], axis=-1) for r in Rs_tr]
    ls_coeff_tr = [np.stack([np.diag(r[:, :, i]) for i in range(0, r.shape[2])], axis=-1) for r in R_flat_tr]
    L_flat_tr = [np.stack([np.linalg.cholesky(r[:, :, i]) for i in range(0, r.shape[2])], axis=-1) for r in R_flat_tr]
    print('Training Started')
    for jj in range(args.epochs):
        '''Generating Task data at every epoch, a seed is used to set a maximum N_ex of samples per task'''
        X,  Y, M, R = batch_data_gen_quantized(int(args.batch_size*n_it_per_epoch), args=args, seed_example=int(jj%N_ex),cov_mats=Rs_tr,L_mats=L_flat_tr,ls_coeffs=ls_coeff_tr)
        if args.large_scale_ICL:
            x_tr, y_tr = LS_ICL_token_processing(X, Y, M, R, args)
        else:
            x_tr, y_tr = ICL_token_processing(X, Y, M, args)
        running_loss=0
        for ii in range(int(np.floor(y_tr.shape[0]/args.batch_size))):
            x_batch=x_tr[ii*args.batch_size:(ii+1)*args.batch_size]
            y_batch=y_tr[ii*args.batch_size:(ii+1)*args.batch_size]
            loss, output = train_step(model_GPT2, ys_batch=y_batch, xs_batch=x_batch, optimizer=optimizer_model_GPT2)
            running_loss = running_loss+loss/int(np.floor(y_tr.shape[0]/args.batch_size))
        if jj%log_every == 0:
            val_loss = val_step(model_GPT2, ys_batch=y_val, xs_batch=x_val)
            print('Validation Accuracy:' + str(val_loss))
            if val_loss<best_val:
                if args.modulationDiversity:
                    torch.save(model_GPT2,args.model_directory+'/modAware_'+ str(args.modulationAware) +'_nUEAware_'+ str(args.numberUEAware) +'_numofAP_' + str(args.numofAP) + '_numofUEmin_' + str(args.numofUE_min) + '_numofUEmax_' + str(args.numofUE_max) + '_modulationScheme_Diversity_tasks_' + str(args.n_tasks) + '_NoisePowerdB_' + str(args.noise_power_dB) + '_bits_'+ str(args.bits) + '.pth')
                else:
                    torch.save(model_GPT2,args.model_directory+'/modAware_'+ str(args.modulationAware) +'_nUEAware_'+ str(args.numberUEAware) +'_numofAP_' + str(args.numofAP) + '_numofUEmin_' + str(args.numofUE_min) + '_numofUEmax_' + str(args.numofUE_max) + '_modulationScheme_'+str(args.modulationScheme)+'_tasks_' + str(args.n_tasks) + '_NoisePowerdB_' + str(args.noise_power_dB) + '_bits_'+ str(args.bits) + '.pth')
                best_val=val_loss
                print('Epoch : '+str(jj)+' -- New Best Model with Validation Accuracy:' + str(val_loss))

