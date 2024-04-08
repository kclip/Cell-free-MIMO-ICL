import argparse

def parameter_reading():
    parser = argparse.ArgumentParser(description="Parameters to train the ICL-Equalizer")
    # -------------------------------------Transformer Params----------------------------------------#
    parser.add_argument('--embedding_dim', type=int, default=64, help='Input embedding dim')
    parser.add_argument('--embedding_dim_single', type=int, default=64,help='Input embedding dim of single layer attention')
    parser.add_argument('--num_head', type=int, default=4, help='Num heads of self attention')
    parser.add_argument('--num_layer', type=int, default=4, help='Transformer layers')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of Transformer')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size of Transformer')
    parser.add_argument('--val_data_size', type=int, default=1000, help='test_data_size')
    parser.add_argument('--epochs', type=int, default=20000, help='number of training epochs')
    parser.add_argument('--n_tasks', default=8192, help='number of pretraining tasks')
    # -------------------------------------Communication System Params----------------------------------------#
    parser.add_argument('--bits', type=int, default=8, help='number of bits for quantization')
    parser.add_argument('--dimPilotMatrix', type=int, default=8, help='the dimension of pilot matrix')
    parser.add_argument('--numofDataSymbols', type=int, default=1, help='number of data symbols')
    parser.add_argument('--numofAnt', type=int, default=2, help='the num of antennas in one AP')
    parser.add_argument('--numofUE_min', type=int, default=1, help='the min num of UEs')
    parser.add_argument('--numofUE_max', type=int, default=4, help='the max num of UEs')
    parser.add_argument('--numofAP', type=int, default=4, help='the num of APs')
    parser.add_argument('--modulationDiversity',  default=True, help='UEs data is generated using different modulations')
    parser.add_argument('--modulationScheme',  default='64QAM', help='If modulation diversity is False, choose between {4QAM 16QAM 64QAM 2PSK 8PSK}')
    parser.add_argument('--modulationAware', default=True, help='Modulation Scheme is provided as input to the model')
    parser.add_argument('--numberUEAware', default=False, help='The equalizer is aware of the number of active UEs')
    parser.add_argument('--large_scale_ICL',default=True, help='ICL using Large Scale Information')
    parser.add_argument('--pilot_contamination', default=True, help='Pilot can be reused if True')
    parser.add_argument('--noise_power_dB',default=-50,help='Noise Power in dB')
    args = parser.parse_args()
    return args