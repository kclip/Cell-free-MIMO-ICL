import numpy as np
import matplotlib.pyplot as plt
import pickle

n_te_setup=1000
N_UE=[4]
colors=['tab:red','tab:blue','tab:green','tab:grey']
markers=['o','x','s','^']
lss=['-.','--',':','-']
Avg_Pathloss=-27
b=8
plt.rcParams["figure.figsize"] = (14, 5)
plt.rc('font', family='serif', serif='Computer Modern Roman', size=13)
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts,amsmath} '
})
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
with open('Results/PilotReuseLSBits_'+str(b)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
Pow_N_TE=loaded_dict['POWS']
ICL_Div=loaded_dict['ICL']
Centr=loaded_dict['Centr']
Centr_u=loaded_dict['Centr_u']
with open('Results/PilotReuseBits_'+str(b)+'.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
ICL_Div_noLS=loaded_dict['ICL']
n_reuse=0
ax1.errorbar(-Pow_N_TE+Avg_Pathloss, Centr_u[n_reuse][:, 0], yerr=Centr_u[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[3], linestyle=lss[3], color=colors[3], label=r'LMMSE $b=\infty$')
ax1.errorbar(-Pow_N_TE+Avg_Pathloss, Centr[n_reuse][:, 0], yerr=Centr[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[0], linestyle=lss[0], color=colors[0], label=r'LMMSE')
ax1.errorbar(-Pow_N_TE+Avg_Pathloss, ICL_Div[n_reuse][:, 0], yerr=ICL_Div[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[2], linestyle=lss[2], color=colors[2], label=r'ICL')
ax1.errorbar(-Pow_N_TE+Avg_Pathloss, ICL_Div_noLS[n_reuse][:, 0], yerr=ICL_Div_noLS[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[1], linestyle=lss[1], color='tab:blue', label=r'ICL without LS tokens')
ax1.set_xlabel(r"Average SNR [dB]")
ax1.set_ylabel(r"Mean squared error (MSE) [dB]")
ax1.set_title('Pilot Reuse : '+str(n_reuse))
ax1.grid()
ax1.set_yscale('log')
ax1.set_xlim(np.min(-Pow_N_TE)+Avg_Pathloss, np.max(-Pow_N_TE)+Avg_Pathloss)
ax1.set_ylim([0.00003, 1.1])

n_reuse=1
ax2.errorbar(-Pow_N_TE+Avg_Pathloss, Centr_u[n_reuse][:, 0], yerr=Centr_u[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[3], linestyle=lss[3], color=colors[3], label=r'LMMSE $b=\infty$')
ax2.errorbar(-Pow_N_TE+Avg_Pathloss, Centr[n_reuse][:, 0], yerr=Centr[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[0], linestyle=lss[0], color=colors[0], label=r'LMMSE')
ax2.errorbar(-Pow_N_TE+Avg_Pathloss, ICL_Div[n_reuse][:, 0], yerr=ICL_Div[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[2], linestyle=lss[2], color=colors[2], label=r'ICL')
ax2.errorbar(-Pow_N_TE+Avg_Pathloss, ICL_Div_noLS[n_reuse][:, 0], yerr=ICL_Div_noLS[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)),marker=markers[1], linestyle=lss[1], color='tab:blue', label=r'ICL without LS tokens')
ax2.set_xlabel(r"Average SNR [dB]")
ax2.set_title('Pilot Reuse : '+str(n_reuse))
ax2.grid()
ax2.set_yscale('log')
ax2.set_xlim(np.min(-Pow_N_TE)+Avg_Pathloss, np.max(-Pow_N_TE)+Avg_Pathloss)
ax2.set_ylim([0.08, 1.1])

n_reuse=2
ax3.errorbar(-Pow_N_TE+Avg_Pathloss, Centr_u[n_reuse][:, 0], yerr=Centr_u[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[3], linestyle=lss[3], color=colors[3], label=r'LMMSE $b=\infty$')
ax3.errorbar(-Pow_N_TE+Avg_Pathloss, Centr[n_reuse][:, 0], yerr=Centr[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[0], linestyle=lss[0], color=colors[0], label=r'LMMSE')
ax3.errorbar(-Pow_N_TE+Avg_Pathloss, ICL_Div[n_reuse][:, 0], yerr=ICL_Div[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[2], linestyle=lss[2], color=colors[2], label=r'ICL')
ax3.errorbar(-Pow_N_TE+Avg_Pathloss, ICL_Div_noLS[n_reuse][:, 0], yerr=ICL_Div_noLS[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[1], linestyle=lss[1], color='tab:blue', label=r'ICL without LS tokens')
ax3.set_xlabel(r"Average SNR [dB]")
ax3.set_title('Pilot Reuse : '+str(n_reuse))
ax3.grid()
ax3.set_yscale('log')
ax3.set_xlim(np.min(-Pow_N_TE)+Avg_Pathloss, np.max(-Pow_N_TE)+Avg_Pathloss)
ax3.set_ylim([0.08, 1.1])


n_reuse=3
ax4.errorbar(-Pow_N_TE+Avg_Pathloss, Centr_u[n_reuse][:, 0], yerr=Centr_u[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[3], linestyle=lss[3], color=colors[3], label=r'LMMSE $b=\infty$')
ax4.errorbar(-Pow_N_TE+Avg_Pathloss, Centr[n_reuse][:, 0], yerr=Centr[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[0], linestyle=lss[0], color=colors[0], label=r'LMMSE')
ax4.errorbar(-Pow_N_TE+Avg_Pathloss, ICL_Div[n_reuse][:, 0], yerr=ICL_Div[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)), marker=markers[2], linestyle=lss[2], color=colors[2], label=r'ICL')
ax4.errorbar(-Pow_N_TE+Avg_Pathloss, ICL_Div_noLS[n_reuse][:, 0], yerr=ICL_Div_noLS[n_reuse][:, 1] / np.sqrt(n_te_setup * len(N_UE)),marker=markers[1], linestyle=lss[1], color='tab:blue', label=r'ICL without LS tokens')
ax4.set_xlabel(r"Average SNR [dB]")
ax4.set_title('Pilot Reuse : '+str(n_reuse))
ax4.grid()
ax4.legend(loc='lower right')
ax4.set_yscale('log')
ax4.set_xlim(np.min(-Pow_N_TE)+Avg_Pathloss, np.max(-Pow_N_TE)+Avg_Pathloss)
ax4.set_ylim([0.08, 1.1])

ax1.set_xticks(np.arange(-15, 35+1, 10.0))
ax2.set_xticks(np.arange(-15, 35 + 1, 10.0))
ax3.set_xticks(np.arange(-15, 35+1, 10.0))
ax4.set_xticks(np.arange(-15, 35 + 1, 10.0))

plt.tight_layout()
plt.savefig('Figures/PilotContamination_' + str(b) + '.png',dpi=400)
plt.clf()


