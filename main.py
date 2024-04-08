import numpy as np
from train import trainNetwork
import torch
import os
from parameters import parameter_reading
from models import build_model

'''Getting Cuda Devices if Available'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found Device: '+str(device))
'''Reading Simulation Parameters'''
args = parameter_reading()
'''Setting Important simulation parameters here (equivalently can be done changing the parameters.py file'''
args.large_scale_ICL=True
args.modulationDiversity=True
args.modulationAware=True
args.numberUEAware=False
args.epochs=30000
args.pilot_contamination=True
'''Training different models for different quantization levels'''
BITs=[1,2,3,4,5,6,7,8]
'''Running Simulations'''
for b in BITs:
    args.bits=b
    args.quantization_levels=np.load("lloyd_codebooks.npz",allow_pickle=True)['arr_0'][int(args.bits-1)]
    print(args)
    if args.large_scale_ICL:
        args.model_directory = './LS_ICL_saved_model_n_layers_' + str(args.num_layer)
    else:
        args.model_directory = './ICL_saved_model_n_layers_' + str(args.num_layer)
    model_path = args.model_directory+'/modAware_'+ str(args.modulationAware) +'_nUEAware_'+ str(args.numberUEAware) +'_numofAP_' + str(args.numofAP) + '_numofUEmin_' + str(args.numofUE_min) + '_numofUEmax_' + str(args.numofUE_max) + '_modulationScheme_Diversity_tasks_' + str(args.n_tasks) + '_NoisePowerdB_' + str(args.noise_power_dB) + '_bits_'+ str(b) + '.pth'
    if not os.path.exists(args.model_directory):
        print('Model Directory Not Found. Creating One.')
        os.makedirs(args.model_directory)
    if os.path.isfile(model_path):
        model = torch.load(model_path, map_location=torch.device(device))
        print('Loading Model From Check-point')
    else:
        model = build_model(embedding_dim=args.embedding_dim, n_positions=args.dimPilotMatrix + args.numofDataSymbols + 1, num_heads=args.num_head, num_layers=args.num_layer, args=args).to(device)
        print('Initializing Model')
    model = build_model(embedding_dim=args.embedding_dim, n_positions=args.dimPilotMatrix+args.numofDataSymbols+1, num_heads=args.num_head, num_layers=args.num_layer,args=args).to(device)
    test_loss = trainNetwork(model.to(device), args)

