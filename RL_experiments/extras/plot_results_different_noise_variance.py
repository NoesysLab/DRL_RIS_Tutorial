import json
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns


var_0 = {
    "avg_score": 4.538992535114288,
    "std_return": 0.5749502182218574,
    "random_policy_average_return": 3.39472827231884,
    "score_as_percentage_of_random": 33.70709438295734,
    "score_as_percentage_of_optimal": 86.80563792119972
}


var_0_1 = {
    "avg_score": 4.435692811489106,
    "std_return": 0.5492310628742825,
    "random_policy_average_return": 3.399346615433693,
    "score_as_percentage_of_random": 30.48662914661895,
    "score_as_percentage_of_optimal": 85.38129352438342
}


var_0_01 = {
    "avg_score": 4.48890149307251,
    "std_return": 0.5306977656858497,
    "random_policy_average_return": 3.3203167489767074,
    "score_as_percentage_of_random": 35.19497784227816,
    "score_as_percentage_of_optimal": 86.18209017145074
}

var_0_001 = {
    "avg_score": 4.41071458864212,
    "std_return": 0.5837245118951166,
    "random_policy_average_return": 3.3868699093461037,
    "score_as_percentage_of_random": 30.229820061015822,
    "score_as_percentage_of_optimal": 84.59243067667131
}

var_1 = {
    "avg_score": 4.210097915649414,
    "std_return": 0.619055432157285,
    "random_policy_average_return": 3.427332845568657,
    "score_as_percentage_of_random": 22.838898506540637,
    "score_as_percentage_of_optimal": 80.76005019268418
}

var_10 = {
    "avg_score": 4.472829722881317,
    "std_return": 0.5435452183361461,
    "random_policy_average_return": 3.3696444489359854,
    "score_as_percentage_of_random": 32.73892218194352,
    "score_as_percentage_of_optimal": 85.92973766206981
}



variances         = np.array([0, 0.001, 0.01, 0.1, 1, 10])
scores_list       = [var_0, var_0_001, var_0_01, var_0_1, var_1, var_10]
bandit_scores     = [scores['avg_score'] for scores in scores_list]
random_scores     = [scores['random_policy_average_return'] for scores in scores_list]
exhaustive_scores = [scores['avg_score'] / (scores["score_as_percentage_of_optimal"]/100) for scores in scores_list]



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


# plt.figure()
# for i, name in enumerate(['Deep RIS Setting', 'Top random', 'Average', 'Exhaustive']):
#
#     plt.plot(df['SNR'], df[name], c=colors[i], ls=line_styles[i], label=r"$\text{"+name+"}$", rasterized=False)
#     plt.scatter(df['SNR'], df[name], c=colors[i], marker=marker_styles[i])



plt.figure()
fig,ax = plt.subplots()

ax.plot(variances, exhaustive_scores, c=colors[2], ls=line_styles[2], label=r'$\text{Optimal policy}$', rasterized=False)
ax.scatter(variances, exhaustive_scores, c=colors[2], marker=marker_styles[2])

ax.plot(variances, bandit_scores, c=colors[0], ls=line_styles[0], label=r'$\text{Neural }\epsilon\text{-greedy}$', rasterized=False)
ax.scatter(variances, bandit_scores, c=colors[0], marker=marker_styles[0])

ax.plot(variances, random_scores, c=colors[1], ls=line_styles[1], label=r'$\text{Random policy}$', rasterized=False)
ax.scatter(variances, random_scores, c=colors[1], marker=marker_styles[1])



ax.set_ylabel(r"$\text{sum-rate} \ ({\rm bps/Hz})$")
ax.set_xlabel(r"$\text{Observation noise variance}$")

ax.set_xscale('log')

ax.legend()
ax.grid()

plt.savefig('./results/plots/sum-rate-varying-observation-noise.pdf')
plt.show()



