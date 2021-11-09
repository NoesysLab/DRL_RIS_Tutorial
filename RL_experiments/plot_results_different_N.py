import json
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns


performances    = dict()
performances[2] = []
performances[2].append({
    "avg_score": 4.208141627311707,
    "std_return": 0.6633358488011213,
    "random_policy_average_return": 2.799872713536024,
    "score_as_percentage_of_random": 50.29760485065577,
    "score_as_percentage_of_optimal": 98.76555290899232
})
performances[2].append({
    "avg_score": 4.262119686603546,
    "std_return": 0.7410011806719987,
    "random_policy_average_return": 2.8348367601037023,
    "score_as_percentage_of_random": 50.347975819518844,
    "score_as_percentage_of_optimal": 97.98017930859572
})
performances[2].append({
    "avg_score": 4.253085828781128,
    "std_return": 0.673462975252015,
    "random_policy_average_return": 2.8606416762769222,
    "score_as_percentage_of_random": 48.67593743220746,
    "score_as_percentage_of_optimal": 98.71429746491503
})
performances[2].append({
    "avg_score": 4.235652259349823,
    "std_return": 0.6653202496580295,
    "random_policy_average_return": 2.759318685129285,
    "score_as_percentage_of_random": 53.503554416418076,
    "score_as_percentage_of_optimal": 99.4827401310922
})
performances[2].append({
    "avg_score": 4.271895527839661,
    "std_return": 0.6908625397330022,
    "random_policy_average_return": 2.802016323298216,
    "score_as_percentage_of_random": 52.457910124209036,
    "score_as_percentage_of_optimal": 98.5545691368784
})


performances[4] = []
performances[4].append({
    "avg_score": 4.941318581104278,
    "std_return": 0.7431195789609486,
    "random_policy_average_return": 3.163140692949295,
    "score_as_percentage_of_random": 56.215580044181344,
    "score_as_percentage_of_optimal": 100.34850354434013
})
performances[4].append({
    "avg_score": 4.938404116630554,
    "std_return": 0.785207870695995,
    "random_policy_average_return": 3.181210160255432,
    "score_as_percentage_of_random": 55.236651081047405,
    "score_as_percentage_of_optimal": 98.57459751764328
})
performances[4].append({
    "avg_score": 3.612474692106247,
    "std_return": 0.699385511221987,
    "random_policy_average_return": 3.1623201265335084,
    "score_as_percentage_of_random": 14.234946101620395,
    "score_as_percentage_of_optimal": 72.62340556453914
})
performances[4].append({
    "avg_score": 4.929538381099701,
    "std_return": 0.763875636995912,
    "random_policy_average_return": 3.1751677668094636,
    "score_as_percentage_of_random": 55.25284782205697,
    "score_as_percentage_of_optimal": 98.88091148326849
})
performances[4].append({
    "avg_score": 3.617553595542908,
    "std_return": 0.6454692334752308,
    "random_policy_average_return": 3.1539753341674803,
    "score_as_percentage_of_random": 14.698220888204672,
    "score_as_percentage_of_optimal": 73.00202086622372
})

performances[4].append({
    "avg_score": 3.6426311898231507,
    "std_return": 0.7210753146752389,
    "random_policy_average_return": 3.202508638302485,
    "score_as_percentage_of_random": 13.743055873658982,
    "score_as_percentage_of_optimal": 73.56194733013805
})


performances[4].append({
    "avg_score": 4.819911598364512,
    "std_return": 0.8262441285782577,
    "random_policy_average_return": 3.175923111438751,
    "score_as_percentage_of_random": 51.76411484914709,
    "score_as_percentage_of_optimal": 95.88887025540713
})


performances[6] = []
performances[6].append({
    "avg_score": 4.538992535114288,
    "std_return": 0.5749502182218574,
    "random_policy_average_return": 3.39472827231884,
    "score_as_percentage_of_random": 33.70709438295734,
    "score_as_percentage_of_optimal": 86.80563792119972
})
performances[6].append({
    "avg_score": 4.118778723716736,
    "std_return": 0.7552909010399604,
    "random_policy_average_return": 2.866404233813286,
    "score_as_percentage_of_random": 43.69148200138433,
    "score_as_percentage_of_optimal": 83.81359386116878
})
performances[6].append({
    "avg_score": 3.831052038192749,
    "std_return": 0.6749010823373882,
    "random_policy_average_return": 2.830493571639061,
    "score_as_percentage_of_random": 35.3492576905692,
    "score_as_percentage_of_optimal": 78.94893716334053
})
performances[6].append({
    "avg_score": 3.923905124664307,
    "std_return": 0.6772638670991784,
    "random_policy_average_return": 2.8705358173549174,
    "score_as_percentage_of_random": 36.695912342944624,
    "score_as_percentage_of_optimal": 80.65749473837936
})
performances[6].append({
    "avg_score": 4.138395951747894,
    "std_return": 0.7941878608247985,
    "random_policy_average_return": 2.827997651517391,
    "score_as_percentage_of_random": 46.33661203811097,
    "score_as_percentage_of_optimal": 84.47973836370744
})

performances[8] = []
performances[8].append({
    "avg_score": 5.046651153087616,
    "std_return": 0.6996665058389814,
    "random_policy_average_return": 2.7697397484779356,
    "score_as_percentage_of_random": 82.20669129151645,
    "score_as_percentage_of_optimal": 90.45798633169568
})
performances[8].append({
    "avg_score": 4.6767824258804325,
    "std_return": 0.7236634253951431,
    "random_policy_average_return": 2.7807444273233415,
    "score_as_percentage_of_random": 68.18454727183104,
    "score_as_percentage_of_optimal": 84.1746130452006
})
performances[8].append({
    "avg_score": 5.011884967486064,
    "std_return": 0.7580649303800867,
    "random_policy_average_return": 2.9137159164746604,
    "score_as_percentage_of_random": 72.01007617619781,
    "score_as_percentage_of_optimal": 90.44627180267977
})
performances[8].append({
    "avg_score": 5.035058653354644,
    "std_return": 0.7016405533908203,
    "random_policy_average_return": 2.854078157345454,
    "score_as_percentage_of_random": 76.41628490082051,
    "score_as_percentage_of_optimal": 90.03376185560198
})
performances[8].append({
    "avg_score": 4.933621516227722,
    "std_return": 0.6779225461509657,
    "random_policy_average_return": 2.761477081576983,
    "score_as_percentage_of_random": 78.65878913651181,
    "score_as_percentage_of_optimal": 88.79846693034479
})



performances[10] = []
performances[10].append({
    "avg_score": 4.918976594924927,
    "std_return": 0.7384186586942427,
    "random_policy_average_return": 2.6450381132364273,
    "score_as_percentage_of_random": 85.96997034973322,
    "score_as_percentage_of_optimal": 86.76782339477728
})
performances[10].append({
    "avg_score": 5.0138448462486265,
    "std_return": 0.7218418088475019,
    "random_policy_average_return": 2.629840107560158,
    "score_as_percentage_of_random": 90.65207925892635,
    "score_as_percentage_of_optimal": 88.55607789187539
})
performances[10].append({
    "avg_score": 4.352527966022492,
    "std_return": 0.730889790908256,
    "random_policy_average_return": 2.57037920665741,
    "score_as_percentage_of_random": 69.33407937432843,
    "score_as_percentage_of_optimal": 77.63183131667373
})
performances[10].append({
    "avg_score": 4.980134633541107,
    "std_return": 0.7051030144587153,
    "random_policy_average_return": 2.592432298898697,
    "score_as_percentage_of_random": 92.10278454163455,
    "score_as_percentage_of_optimal": 88.24840259080652
})
performances[10].append({
    "avg_score": 4.922305210431417,
    "std_return": 0.7021700419141577,
    "random_policy_average_return": 2.6430421963334085,
    "score_as_percentage_of_random": 86.23634602807108,
    "score_as_percentage_of_optimal": 87.1556045861064
})





N_controllable         = np.array([2,4,6,8,10])
random_results_avg     = np.zeros_like(N_controllable, dtype=float)
random_results_std     = np.zeros_like(N_controllable, dtype=float)
exhaustive_results_avg = np.zeros_like(N_controllable, dtype=float)
exhaustive_results_std = np.zeros_like(N_controllable, dtype=float)
bandit_results_avg     = np.zeros_like(N_controllable, dtype=float)
bandit_results_std     = np.zeros_like(N_controllable, dtype=float)

bandit_avg_norm        = np.zeros_like(N_controllable, dtype=float)
bandit_std_norm        = np.zeros_like(N_controllable, dtype=float)
random_avg_norm        = np.zeros_like(N_controllable, dtype=float)
random_std_norm        = np.zeros_like(N_controllable, dtype=float)


for i,N in enumerate(N_controllable):
    curr_exhaustive_perf = []
    curr_random_perf     = []
    curr_bandit_perf     = []

    for scores in performances[N]:
        curr_bandit_perf.append(scores["avg_score"])
        curr_random_perf.append(scores["random_policy_average_return"])
        curr_exhaustive_perf.append(scores["avg_score"]/(scores["score_as_percentage_of_optimal"]/100))

    curr_exhaustive_perf = np.array(curr_exhaustive_perf)
    curr_random_perf     = np.array(curr_random_perf)
    curr_bandit_perf     = np.array(curr_bandit_perf)

    random_results_avg[i]     = curr_random_perf.mean()
    random_results_std[i]     = curr_random_perf.std()
    exhaustive_results_avg[i] = curr_exhaustive_perf.mean()
    exhaustive_results_std[i] = curr_exhaustive_perf.std()
    bandit_results_avg[i]     = curr_bandit_perf.mean()
    bandit_results_std[i]     = curr_bandit_perf.std()

    bandit_avg_norm[i]        = (curr_bandit_perf / curr_exhaustive_perf).mean()
    bandit_std_norm[i]        = (curr_bandit_perf / curr_exhaustive_perf).std()
    random_avg_norm[i]        = (curr_random_perf / curr_exhaustive_perf).mean()
    random_std_norm[i]        = (curr_random_perf / curr_exhaustive_perf).std()




colors = sns.color_palette("Set1")
line_styles = ['-','--', ':', '-.', '.']
marker_styles = ["o", "s", 'x', 'D']
hatches = ['/', '-', '\\', '|', '+', 'x', 'o', 'O', '.', '*']

sns.set_style('ticks')
from matplotlib import rc
rc('text', usetex=True)
fontsize = 16
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["savefig.dpi"] = 400
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.0
plt.rcParams["savefig.transparent"] = True
plt.rcParams["axes.labelsize"] = fontsize + 2
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}\usepackage{amsmath}'
#plt.rcParams['font.size'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize -2




fig,ax = plt.subplots()

ax.plot(N_controllable, exhaustive_results_avg, c=colors[2], ls=line_styles[2], label=r'$\text{Optimal policy}$', rasterized=False)
ax.scatter(N_controllable, exhaustive_results_avg, c=colors[2], marker=marker_styles[2])
#ax.fill_between(N_controllable, exhaustive_results_avg-exhaustive_results_std, exhaustive_results_avg+exhaustive_results_std, color=colors[2], alpha=0.2)

ax.plot(N_controllable, bandit_results_avg, c=colors[0], ls=line_styles[0], label=r'$\text{Neural }\epsilon\text{-greedy}$', rasterized=False)
ax.scatter(N_controllable, bandit_results_avg, c=colors[0], marker=marker_styles[0])
#ax.fill_between(N_controllable, bandit_results_avg-bandit_results_std, bandit_results_avg+bandit_results_std, color=colors[0], alpha=0.2)

ax.plot(N_controllable, random_results_avg, c=colors[1], ls=line_styles[1], label=r'$\text{Random policy}$', rasterized=False)
ax.scatter(N_controllable, random_results_avg, c=colors[1], marker=marker_styles[1])
#ax.fill_between(N_controllable, random_results_avg-random_results_std, random_results_avg+random_results_std, color=colors[1], alpha=0.2)



ax.set_ylabel(r"${\rm Sum\mbox{-}Rate~~(bps/Hz)}$")




ax.set_xlabel(r"${\rm Total~RIS~meta\mbox{-}atoms}~(N_{{\rm tot}})$")
ax.set_xticklabels([None, '$32$', '$48$', '$64$', '$80$', '$160$', None])




ax2 = ax.twiny()
ax2.set_xticks( ax.get_xticks() )
ax2.set_xbound(ax.get_xbound())
ax2.set_xlabel(r"${\rm Number~of~actions~}({\rm card}(\mathcal{A}))$")
ax2.set_xticklabels([None, '$16$', '$64$', '$256$', '$1024$', '$4096$', None])

ax.legend()
ax.grid()

plt.savefig('./results/plots/sum-rate-varying-N-plot.pdf')
plt.show()



ind = np.arange(len(N_controllable))
fig,ax = plt.subplots()
width = 0.3

plt.bar(ind, bandit_avg_norm, width, color=colors[0], label=r'$\text{Neural }\epsilon\text{-greedy}$', rasterized=False)
plt.bar(ind+width, random_avg_norm, width, color=colors[1], label=r'$\text{Random policy}$', hatch=hatches[0], rasterized=False)



ax.set_ylabel(r"${\rm Normalized~Sum\mbox{-}Rate}$")
ax.set_ylim([0.3, 1.01])


ax.set_xticks(ind + width / 2)
ax.set_xlabel(r"${\rm Total~RIS~meta\mbox{-}atoms}~(N_{{\rm tot}})$")
ax.set_xticklabels(['$32$', '$48$', '$64$', '$80$', '$160$'])




ax2 = ax.twiny()
ax2.set_xticks( ax.get_xticks() )
ax2.set_xbound(ax.get_xbound())
ax2.set_xlabel(r"${\rm Number~of~actions~}({\rm card}(\mathcal{A}))$")
ax2.set_xticklabels(['$16$', '$64$', '$256$', '$1024$', '$4096$'])

ax.legend()
ax.grid()

plt.savefig('./results/plots/sum-rate-varying-N-bar.pdf')
plt.show()

