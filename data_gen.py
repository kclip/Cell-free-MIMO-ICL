import numpy as np
import  scipy
from scipy.integrate import quad
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

def get_data_symbol(args,numofUE=1,symbols=None):
    '''
    Used to generate data symbols
    :param args: Simulation parameters
    :param numofUE: number of UEs
    :param symbols: if one wants to get specific constellation points from data symbols, this is set to a list of symbols.
    :return: The list of complex constellation points, either randomly generated (symbols=None) or corresponding to the symbols specified, and the modulation indicator.
    '''
    if args.modulationDiversity:
        args.modulationScheme = np.random.choice(['4QAM', '16QAM', '64QAM', '2PSK', '8PSK'])
    if args.modulationScheme == '4QAM':
        mod_id = 0./4.
        if symbols is None:
            symbols = np.random.randint(0, 4, (numofUE, args.numofDataSymbols))
        x = (symbols % 2 - 1 / 2) + 1j * (np.floor(symbols / 2) - 1 / 2)
        x = x / np.sqrt(0.5000000000000001)
    elif args.modulationScheme == '16QAM':
        mod_id = 1./4.
        if symbols is None:
            symbols=np.random.randint(0,16,(numofUE,args.numofDataSymbols))
        x = (symbols % 4 - 3 / 2) + 1j * (np.floor(symbols / 4) - 3 / 2)
        x = x / np.sqrt(2.5000000000000004)
    elif args.modulationScheme == '64QAM':
        mod_id = 2./4.
        if symbols is None:
            symbols = np.random.randint(0, 64,(numofUE,args.numofDataSymbols))
        x = (symbols % 8 - 7 / 2) + 1j * (np.floor(symbols / 8) - 7 / 2)
        x = x / np.sqrt(10.5)
    elif args.modulationScheme == '2PSK':
        mod_id = 3./4.
        if symbols is None:
            symbols = np.random.randint(0, 2, (numofUE, args.numofDataSymbols))
        angle = 2 * np.pi * symbols / (2.)
        x = np.cos(angle) + 1j * np.sin(angle)
    elif args.modulationScheme == '8PSK':
        mod_id = 4./4.
        if symbols is None:
            symbols = np.random.randint(0, 8,(numofUE,args.numofDataSymbols))
        angle = 2 * np.pi * symbols / (8.)
        x = np.cos(angle) + 1j * np.sin(angle)
    return x, mod_id

def walsh_hadamard_matrix(size):
    '''
    Walsh-Hadamard matrix for pilot generation
    :param size: The length of the pilot sequence
    :return: The Hadamard matrix of dimension = size
    '''
    # Check if the size is a power of 2
    if size & (size - 1) != 0:
        raise ValueError("Size must be a power of 2.")
    # Initialize the matrix for the base case
    if size == 1:
        return np.array([[1]])
    # Generate the smaller matrix
    smaller_matrix = walsh_hadamard_matrix(size // 2)
    # Form the larger matrix from the smaller one
    top = np.hstack((smaller_matrix, smaller_matrix))
    bottom = np.hstack((smaller_matrix, -smaller_matrix))
    return np.vstack((top, bottom)) + 0j

def non_uniform_quantizer(X, q_lvls):
    '''
    Performs non-uniform quantization using the quantization levels (q_lvls) that are specified the file Lloyd_codebooks.npz for bits={1,2,3,4,5,6,7,8}. Quantization is performed directly on the signal X.
    :param X: Input signal
    :param q_lvls: Quantization levels
    :return: Quantized Signal
    '''
    ids_re =np.argmin(np.abs(np.repeat(np.real(X)[:,:,np.newaxis],len(q_lvls),axis=-1)-q_lvls),axis=-1)
    ids_im =np.argmin(np.abs(np.repeat(np.imag(X)[:,:,np.newaxis],len(q_lvls),axis=-1)-q_lvls),axis=-1)
    X_real_q=q_lvls[ids_re]
    X_imag_q = q_lvls[ids_im]
    return X_real_q + 1j * X_imag_q

def non_uniform_quantizer_scaled(X, q_lvls,scaling_coeff):
    '''
     Performs non-uniform quantization using the quantization levels (q_lvls) that are specified the file Lloyd_codebooks.npz for bits={1,2,3,4,5,6,7,8}. Quantization is performed by first scaling the signal the quantization levels using the scaling_coeff.
     :param X: Input signal
     :param q_lvls: Quantization levels
     :return: Quantized Signal
     '''
    q_lvls_ext=np.tile(q_lvls,(X.shape[0],1))*scaling_coeff
    ids_re =np.argmin(np.abs(np.repeat(np.real(X)[:,:,np.newaxis],len(q_lvls),axis=-1)-q_lvls_ext[:,np.newaxis,:]),axis=-1)
    ids_im =np.argmin(np.abs(np.repeat(np.imag(X)[:,:,np.newaxis],len(q_lvls),axis=-1)-q_lvls_ext[:,np.newaxis,:]),axis=-1)
    X_real_q=np.stack([q_lvls_ext[i,ids_re[i,:]] for i in range(0,X.shape[0])])
    X_imag_q = np.stack([q_lvls_ext[i,ids_im[i,:]] for i in range(0,X.shape[0])])
    return X_real_q + 1j * X_imag_q

def batch_data_gen_quantized(batch_size, args, seed_example=0,cov_mats=None,L_mats=None,ls_coeffs=None,TESTING=False,SCALING=True):
    '''
    Generation of the data based on the covariance matrices
    :param batch_size: number of data points to generate
    :param args:  Simulation parameters
    :param seed_example: Random Seed
    :param cov_mats: Task covariance matrices
    :param L_mats: Precomputed Cholesky Factorized Task Covariance matrices to sample random the channels
    :param ls_coeffs: Precomputed Large scale coeffs
    :param TESTING: If True there is no randomness in the sampling of the covariance matrices (good for testing)
    :param SCALING: If True the quantization method used it the "non_uniform_quantizer" otherwise "non_uniform_quantizer_scaled"
    :return: Transmitted pilots and data X_batch, received pilots and data Y_batch, modulation informations of the generated data, Covariance matrices of the generated data
    '''
    np.random.seed(seed_example)
    wh_matrix = walsh_hadamard_matrix(args.dimPilotMatrix)
    '''Sample Channel Vectors from Covariance Matrices'''
    if TESTING:
        id_rand=np.arange(0,len(cov_mats))
    else:
        id_rand=np.random.choice(np.arange(0,len(cov_mats)),size=batch_size,replace=False)
    R_eval=[cov_mats[id] for id in id_rand]
    ls_coeffs=[ls_coeffs[id] for id in id_rand]
    H=generate_channel_realizations([L_mats[id] for id in id_rand], seed_example)
    Hs,X_batch,Y_batch,MODs,Y_err= [],[],[],[],[]
    for ii in range(0,batch_size):
        H_eval = H[ii]
        ls_coeff_eval=np.real(np.sum(ls_coeffs[ii],axis=1))
        Hs.append(H_eval)
        numofUE = H_eval.shape[-1]
        '''Generate data symbols for the UEs from the specified constellation'''
        data, m_id = get_data_symbol(args,numofUE)
        if args.n_reuse>0:
            pilot_assignment = np.random.choice(np.arange(0, args.dimPilotMatrix), numofUE, replace=False)
            pilot_assignment[-args.n_reuse-1:] = pilot_assignment[-1]
        else:
            if args.pilot_contamination:
                pilot_assignment = np.random.choice(np.arange(0, args.dimPilotMatrix), numofUE, replace=True)
            else:
                if args.dimPilotMatrix < numofUE:
                    raise Exception("The num of UE should be less than pilot dimension")
                    sys.exit(1)
                pilot_assignment = np.random.choice(np.arange(0, args.dimPilotMatrix), numofUE, replace=False)
        pilots = wh_matrix[pilot_assignment, :]
        X = np.hstack((pilots, data))
        noise_std = np.sqrt((10 ** (args.noise_power_dB / 10.)))
        n = noise_std * (np.random.randn(args.numofAnt * args.numofAP, X.shape[1]) + 1j * np.random.randn(args.numofAnt * args.numofAP, X.shape[1])) / np.sqrt(2)
        Y = np.matmul(H_eval, X) + n
        if SCALING:
            Y_s = Y / (np.sqrt(ls_coeff_eval)[:, np.newaxis] / np.sqrt(2) + noise_std / np.sqrt(2))
            Y_q =  non_uniform_quantizer( Y_s ,args.quantization_levels)
        else:
            Y_q = non_uniform_quantizer_scaled(Y, args.quantization_levels,(np.sqrt(ls_coeff_eval)[:, np.newaxis] / np.sqrt(2) + noise_std / np.sqrt(2)))
        MODs.append(np.ones((args.dimPilotMatrix + 1, 1)) * m_id)
        X_batch.append(np.transpose(X))
        Y_batch.append(np.transpose(Y_q))
    return X_batch, Y_batch, MODs, R_eval

def batch_data_gen_unquantized(batch_size, args, seed_example=0,cov_mats=None,L_mats=None,TESTING=False):
    '''
       Generation of the data based on the covariance matrices
       :param batch_size: number of data points to generate
       :param args:  Simulation parameters
       :param seed_example: Random Seed
       :param cov_mats: Task covariance matrices
       :param L_mats: Precomputed Cholesky Factorized Task Covariance matrices to sample random the channels
       :param ls_coeffs: Precomputed Large scale coeffs
       :param TESTING: If True there is no randomness in the sampling of the covariance matrices (good for testing)
       :return: Transmitted pilots and data X_batch, received pilots and data Y_batch, modulation informations of the generated data, Covariance matrices of the generated data
       '''
    '''Empirical Mean and Std of received vector entries'''
    np.random.seed(seed_example)
    wh_matrix = walsh_hadamard_matrix(args.dimPilotMatrix)
    '''Sample Channel Vectors from Covariance Matrices'''
    if TESTING:
        id_rand=np.arange(0,len(cov_mats))
    else:
        id_rand=np.random.choice(np.arange(0,len(cov_mats)),size=batch_size,replace=False,)
    H = generate_channel_realizations([L_mats[id] for id in id_rand], seed_example)
    Hs,X_batch,Y_batch,MODs = [],[],[],[]

    for ii in range(0,batch_size):
        H_eval = H[ii]
        Hs.append(H_eval)
        numofUE = H_eval.shape[-1]
        '''Generate data symbols for the UEs from the specified constellation'''
        data, m_id = get_data_symbol(args,numofUE)
        if args.n_reuse > 0:
            pilot_assignment = np.random.choice(np.arange(0, args.dimPilotMatrix), numofUE, replace=False)
            pilot_assignment[-args.n_reuse-1:] = pilot_assignment[-1]
        else:
            if args.pilot_contamination:
                pilot_assignment = np.random.choice(np.arange(0, args.dimPilotMatrix), numofUE, replace=True)
            else:
                if args.dimPilotMatrix < numofUE:
                    raise Exception("The num of UE should be less than pilot dimension")
                    sys.exit(1)
                pilot_assignment = np.random.choice(np.arange(0, args.dimPilotMatrix), numofUE, replace=False)
        pilots = wh_matrix[pilot_assignment, :]
        X = np.hstack((pilots, data))
        noise_std = np.sqrt((10 ** (args.noise_power_dB / 10.)))
        n = noise_std * (np.random.randn(args.numofAnt * args.numofAP, X.shape[1]) + 1j * np.random.randn(args.numofAnt * args.numofAP, X.shape[1])) / np.sqrt(2)
        Y = np.matmul(H_eval, X) + n
        MODs.append(np.ones((args.dimPilotMatrix + 1, 1)) * m_id)
        X_batch.append(np.transpose(X))
        Y_batch.append(np.transpose(Y))
    return X_batch, Y_batch, MODs

def functionRlocalscattering(M, theta, ASDdeg, antennaSpacing=None):
    '''
    Generate the spatial correlation matrix for the local scattering model, defined in (2.23) for different angular distributions.
    Parameters:
    :param M: Number of antennas
    :param theta:  Nominal angle
    :param ASDdeg: Angular standard deviation around the nominal angle (measured in degrees)
    :param antennaSpacing:   Spacing between antennas (in wavelengths). Default is None (0.5 wavelengths).
    :return:  M x M spatial correlation matrix
    '''
    # Set the antenna spacing if not specified by input
    if antennaSpacing is None:
        antennaSpacing = 1. / 2.
    # Compute the ASD in radians based on input
    ASD = ASDdeg * np.pi / 180.
    # The correlation matrix has a Toeplitz structure, so we only need to compute the first row of the matrix
    firstRow = np.zeros(M)+1j*np.zeros(M)
    # Go through all the columns of the first row
    for column in range(1, M + 1):
        # Distance from the first antenna
        distance = antennaSpacing * (column - 1)
        # For Gaussian angular distribution
        def F(Delta):
            return np.exp(1j * 2 * np.pi * distance * np.sin(theta + Delta)) * np.exp(-Delta ** 2 / (2 * ASD ** 2)) / (np.sqrt(2 * np.pi) * ASD)
        # Compute the integral in (2.23) by including 3 standard deviations
        firstRow[column - 1], _ = quad(F, -20 * ASD, 20 * ASD,complex_func=True)
    # Compute the spatial correlation matrix by utilizing the Toeplitz structure
    R= scipy.linalg.toeplitz(firstRow)
    return R

def generate_covariance_matrices(L, K_min,K_max, N, nbrOfSetups,seed):
    '''
    :param L: number of APs
    :param K_min: min number of UEs
    :param K_max: max number of UEs
    :param N: number of antennas per UE
    :param nbrOfSetups: number of tasks
    :param seed: random seed
    :return: List of Channel Covariance Matrices [N,N,L,K] with K between K_min and K_max (included)
    '''
    # Size of coverage area
    np.random.seed(seed)
    squareLength = 1000
    # Communication bandwidth
    B = 20e6
    # Noise figure (in dB)
    noiseFigure = 5
    # Compute noise power
    noiseVariancedBm = -174 + 10 * np.log10(B) + noiseFigure
    # Pathloss parameter
    alpha = 36.7
    constantTerm = -30.5
    # Standard deviation of the shadow fading
    sigma_sf = 4
    # Decorrelation distance of the shadow fading
    decorr = 9
    # Height difference between an AP and a UE
    distanceVertical = 10
    # define the antenna spacing (in number of wavelengths)
    antennaSpacing = 1 / 2
    # Angular standard deviation around the nominal angle (measured in degrees)
    ASDdeg = 15
    # Numbers of APs per dimension on the grid
    nbrAPsPerDim = int(np.sqrt(L))
    # Prepare to save results
    interAPDistance = squareLength / nbrAPsPerDim
    locationsGridHorizontal = np.repeat(np.arange(interAPDistance / 2, squareLength, interAPDistance), nbrAPsPerDim)
    locationsGridVertical = np.tile(np.arange(interAPDistance / 2, squareLength, interAPDistance), nbrAPsPerDim)
    APpositions = locationsGridHorizontal + 1j * locationsGridVertical
    R_APs = []
    mean_gain = []
    for n in range(nbrOfSetups):
        K=np.random.randint(K_min,K_max+1)
        R_AP = np.zeros((N, N, L, K)) + 1j * np.zeros((N, N, L, K))
        gainOverNoisedB_AP = np.zeros((L, K)) + 1j * np.zeros((L, K))
        UEpositions = np.zeros(K, dtype=np.complex128)
        wrapHorizontal = np.tile(np.array([-squareLength, 0, squareLength]), (3, 1))
        wrapVertical = wrapHorizontal.T
        wrapLocations = (wrapHorizontal.T).flatten() + 1j * (wrapVertical.T).flatten()
        shadowCorrMatrix = sigma_sf ** 2 * np.ones((K, K))
        nbrOfUEs = 0
        while nbrOfUEs < K:
            UEposition = np.random.rand() * squareLength + 1j * np.random.rand() * squareLength
            if nbrOfUEs > 0:
                # Compute distances from the new prospective UE to all other UEs
                shortestDistances = np.zeros(nbrOfUEs)
                for i in range(nbrOfUEs):
                    shortestDistances[i] = np.min(np.abs(UEposition - UEpositions[i] + wrapLocations))
                newcolumn = (sigma_sf ** 2) * (2 ** (-shortestDistances / decorr))
            else:
                newcolumn = []
            # Compute and store the UE index
            nbrOfUEs += 1
            k = nbrOfUEs
            # print(" number of UE is:", nbrOfUEs)
            shadowCorrMatrix[:nbrOfUEs - 1, nbrOfUEs-1] = newcolumn
            shadowCorrMatrix[nbrOfUEs-1, :nbrOfUEs - 1] = newcolumn
            # Store the UE position
            UEpositions[k-1] = UEposition
        APpositionsWrapped = APpositions[:, np.newaxis] + wrapLocations
        shadowAPrealizations = np.dot(scipy.linalg.sqrtm(shadowCorrMatrix), np.random.randn(K, L))
        # Go through all UEs
        for k in range(K):
            distanceAPstoUE, whichpos = np.min(np.abs(APpositionsWrapped - UEpositions[k]), axis=1), np.argmin(np.abs(APpositionsWrapped - UEpositions[k]), axis=1)
            distances = np.sqrt(distanceVertical ** 2 + distanceAPstoUE ** 2)
            # Compute the channel gain divided by the noise power (in dB)
            mean_gain.append(constantTerm - alpha * np.log10(distances))
            gainOverNoisedB_AP[:, k] = constantTerm - alpha * np.log10(distances) + shadowAPrealizations[k, :] - noiseVariancedBm
            # Go through all APs
            for l in range(L):
                # Compute nominal angle between UE k and AP l
                angletoUE = np.angle(UEpositions[k] - APpositionsWrapped[l, whichpos[l]])
                # Generate normalized spatial correlation matrix using the local scattering model
                local_scattering=functionRlocalscattering(N, angletoUE, ASDdeg, antennaSpacing)
                R_AP[:, :, l, k] = 10 ** (gainOverNoisedB_AP[l, k] / 10.) * local_scattering
        R_APs.append(R_AP)
    return R_APs

def generate_channel_realizations(L,seed):
    '''
    :param L: Cholesky matrix
    :param seed: random seed
    :return: Samples from Gaussian distribution with covariance matrix with Cholesky factorization L
    '''
    np.random.seed(seed)
    H=[np.stack([np.transpose((np.random.randn(l.shape[0])+1j*np.random.randn(l.shape[0]))/np.sqrt(2))@l[:,:,i] for i in range(0,l.shape[2])],axis=-1) for l in L]
    return H