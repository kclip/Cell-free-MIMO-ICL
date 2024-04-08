import numpy as np
import matplotlib.pyplot as plt
import pickle
with open('Results/ResultsVsBitsNoContaminationLS.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
plt.rcParams["figure.figsize"] = (7,5)
plt.rc('font', family='serif', serif='Computer Modern Roman', size=13)
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts,amsmath} '
})
n_te_setup=1000
N_UE=[1,2,3,4]
colors=['tab:red','tab:blue','tab:green','tab:grey']
markers=['o','x','s','^']
lss=['-.','--',':','-']
BITS_TE=loaded_dict['BITS']
ICL_Div=loaded_dict['ICL']
Centr=loaded_dict['Centr']
Centr_u=loaded_dict['Centr_u']
with open('Results/MAMLBits_8_onlyPilots.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
MAML=loaded_dict['MAML']
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.errorbar(BITS_TE, Centr_u[:, 0], yerr=Centr_u[:, 1] / np.sqrt(n_te_setup * len(N_UE)),marker=markers[3],linestyle=lss[3],color=colors[3], label=r'LMMSE $b=\infty$')
ax1.errorbar(BITS_TE, Centr[:, 0], yerr=Centr[:, 1] / np.sqrt(n_te_setup * len(N_UE)),marker=markers[0],linestyle=lss[0],color=colors[0], label=r'LMMSE')
ax1.errorbar(BITS_TE, ICL_Div[:, 0], yerr=ICL_Div[:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[2], linestyle=lss[2], color=colors[2],   label=r'ICL')
ax1.errorbar(BITS_TE, MAML[0], yerr=ICL_Div[:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[1], linestyle=lss[1], color='tab:blue',   label=r'MAML')
ax = plt.gca()
ax1.set_ylabel(r"Mean squared error (MSE) [dB]")
ax1.set_xlabel(r'Fronthaul Capacity $b$ [bits/symbol]')
ax1.grid()
ax1.set_title('Orthogonal Pilots')
ax1.set_yscale('log')
ax1.set_xlim(np.min(BITS_TE), np.max(BITS_TE))
ax1.set_ylim(0.5*np.min(Centr_u[:, 0]),1)


with open('Results/ResultsVsBitsContaminationLS.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
BITS_TE=loaded_dict['BITS']
ICL_Div=loaded_dict['ICL']
Centr=loaded_dict['Centr']
Centr_u=loaded_dict['Centr_u']
ax2.errorbar(BITS_TE, Centr_u[:, 0], yerr=Centr_u[:, 1] / np.sqrt(n_te_setup * len(N_UE)),marker=markers[3],linestyle=lss[3],color=colors[3], label=r'LMMSE $b=\infty$')
ax2.errorbar(BITS_TE, Centr[:, 0], yerr=Centr[:, 1] / np.sqrt(n_te_setup * len(N_UE)),marker=markers[0],linestyle=lss[0],color=colors[0],label=r'LMMSE')
ax2.errorbar(BITS_TE, ICL_Div[:, 0], yerr=ICL_Div[:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[2], linestyle=lss[2], color=colors[2],   label=r'ICL')
ax2.errorbar(BITS_TE, MAML[1], yerr=ICL_Div[:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[1], linestyle=lss[1], color='tab:blue',   label=r'MAML')
ax2.set_xlabel(r'Fronthaul Capacity $b$ [bits/symbol]')
ax2.grid()
ax2.legend(loc='upper right')
ax2.set_ylim(0.5*np.min(ICL_Div[:, 0]),1)
ax2.set_xlim(np.min(BITS_TE), np.max(BITS_TE))
ax2.set_yscale('log')
ax2.set_title('Pilot Contamination')
plt.tight_layout()
plt.savefig('Figures/MSEvsBITS.png',dpi=400)
plt.clf()
plt.close()

