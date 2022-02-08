import json
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns


N2 = {
    "avg_score": 4.355014866987864,
    "std_return": 0.6970836257902465,
    "random_policy_average_return": 2.9386425186196963,
    "score_as_percentage_of_random": 48.19818468540533,
    "score_as_percentage_of_optimal": 99.67167477730878
}
N4 = {
    "avg_score": 3.553308363755544,
    "std_return": 0.7094866019915426,
    "random_policy_average_return": 3.0757171817620597,
    "score_as_percentage_of_random": 15.527799006535293,
    "score_as_percentage_of_optimal": 70.41462621879973
}
N6 = {
    "avg_score": 3.79752566019694,
    "std_return": 0.7608988171148996,
    "random_policy_average_return": 2.887116892139117,
    "score_as_percentage_of_random": 31.533491786793743,
    "score_as_percentage_of_optimal": 77.13475141256629
}
N8 = {
    "avg_score": 4.3110471256573994,
    "std_return": 0.6654583625766881,
    "random_policy_average_return": 2.849038345317046,
    "score_as_percentage_of_random": 51.315868834951004,
    "score_as_percentage_of_optimal": 77.59013816259943
}
N10 = {
    "avg_score": 3.1516940772533415,
    "std_return": 0.6626915088857651,
    "random_policy_average_return": 2.6087297087907793,
    "score_as_percentage_of_random": 20.813362405193047,
    "score_as_percentage_of_optimal": 55.638080856839856
}


N_controllable      = np.array([2,4,6,8,10])
scores_list         = [N2, N4, N6, N8, N10]
random_scores       = np.array([scores['random_policy_average_return'] for scores in scores_list])
exhaustive_scores   = np.array([scores['avg_score'] / (scores["score_as_percentage_of_optimal"]/100) for scores in scores_list])
UCB_scores_norm     = np.array([scores["score_as_percentage_of_optimal"]/100 for scores in scores_list])
random_scores_norm  = random_scores / exhaustive_scores


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




# ax.plot(N_controllable, UCB_scores_norm, c=colors[3], ls=line_styles[3], label=r'$\text{UCB}$', rasterized=False)
# ax.scatter(N_controllable, UCB_scores_norm, c=colors[3], marker=marker_styles[3])
#
# ax.plot(N_controllable, random_scores_norm, c=colors[1], ls=line_styles[1], label=r'$\text{Random policy}$', rasterized=False)
# ax.scatter(N_controllable, random_scores_norm, c=colors[1], marker=marker_styles[1])
#

ind = np.arange(len(N_controllable))

fig,ax = plt.subplots()
width = 0.3

plt.bar(ind, UCB_scores_norm, width, color=colors[3], label=r'$\text{UCB}$', rasterized=False)
plt.bar(ind+width, random_scores_norm, width, color=colors[1], label=r'$\text{Random policy}$', hatch=hatches[0], rasterized=False)



ax.set_ylabel(r"${\rm Normalized~Sum\mbox{-}Rate}$")
ax.set_ylim([0.3, 1.03])



ax.set_xlabel(r"${\rm Total~RIS~meta\mbox{-}atoms}~(N_{{\rm tot}})$")
ax.set_xticklabels([None, '$32$', '$48$', '$64$', '$80$', '$160$', None])




ax2 = ax.twiny()
ax2.set_xticks( ax.get_xticks() )
ax2.set_xbound(ax.get_xbound())
ax2.set_xlabel(r"${\rm Number~of~actions~}({\rm card}(\mathcal{A}))$")
ax2.set_xticklabels([None, '$16$', '$64$', '$256$', '$1024$', '$4096$', None])

ax.legend()
ax.grid()

plt.savefig('./results/plots/UCB-sum-rate-varying-N.pdf')
plt.show()

