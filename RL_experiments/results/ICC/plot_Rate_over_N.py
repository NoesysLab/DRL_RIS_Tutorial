import re
import os, sys
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

results_path = './'

#
# def get_top_level_dirnames(path):
#     return next(os.walk(path))[1]
#
#
# def find_agent_dirnames(agent_name, setup_dirname):
#     all_agent_dirnames  = get_top_level_dirnames(setup_dirname)
#     this_agent_dirnames = []
#
#     for this_agent_dirname in all_agent_dirnames:
#         re_agent = re.compile(agent_name)
#         m = re.search(re_agent, this_agent_dirname)
#         if m:
#             this_agent_dirnames.append(this_agent_dirname)
#
#     if len(this_agent_dirnames) == 0:
#         print(f" > Not found any directories for {agent_name}")
#     else:
#         print(f" > Found directories for {agent_name}:")
#         for dirname in this_agent_dirnames:
#             print(f"  -  {dirname}")
#
#     return this_agent_dirnames
#
#
# def get_agent_rate_score(setup_dirname, agent_dirname):
#     with open(setup_dirname + "/" + agent_dirname + "/agent_performance.json") as fagent:
#         agent_performance = json.load(fagent)
#         rate = float(agent_performance['avg_score'])
#
#         return rate
#
#
# re_setup_dirname = re.compile(r'setup__N_controllable_\d+_K_\d+_M_\d+_codebook_rays_per_RX_\d+_kappa_H_\d+_observation_noise_variance_\d+')
#
#
#
#
#
# TrialData = namedtuple('TrialData', ['N_total',
#                                      'N_controllable',
#                                      'card_A',
#                                      'optimal_rate',
#                                      'random_rate',
#                                      'DQN_rate',
#                                      'Bandit_rate',
#                                      'UCB_rate'])
#
#
# all_trial_data = []
#
#
# for dirname in get_top_level_dirnames(results_path):
#
#     m = re.match(re_setup_dirname, dirname)
#
#     if m is None:
#         print(f"Not matched: {dirname}")
#         continue
#
#     print(f'Processing trial with dirname "{dirname}"')
#
#
#
#     with open(dirname+'/setup.json') as fsetup:
#         setup_params   = json.load(fsetup)
#         N_total        = int(setup_params['N']) * int(setup_params['M'])
#         N_controllable = int(setup_params['N_controllable'])
#         card_A         = 2 ** int(N_controllable + 2)
#
#     with open(dirname+'/baselines.json') as fbaselines:
#         baselines_scores = json.load(fbaselines)
#         optimal_rate     = float(baselines_scores['optimal_score'])
#         random_rate      = float(baselines_scores['random_policy_average_return'])
#
#
#     DQN_dirnames = find_agent_dirnames('DQN', dirname)
#     DQN_dirname  = DQN_dirnames[0]
#     DQN_rate     = get_agent_rate_score(dirname, DQN_dirname)
#
#
#     Bandit_dirnames = find_agent_dirnames('Neural Softmax', dirname)
#     Bandit_dirname  = Bandit_dirnames[0]
#     Bandit_rate     = get_agent_rate_score(dirname, Bandit_dirname)
#
#     UCB_dirnames    = find_agent_dirnames('UCB', dirname)
#     UCB_dirname     = None
#     re_UCB_dirname  = re.compile("alpha_10")
#     for UCB_dir in UCB_dirnames:
#         m_ = re.search(re_UCB_dirname, UCB_dir)
#         if m_:
#             print(f' + Using "{UCB_dir}" for UCB')
#             UCB_dirname = UCB_dir
#             break
#
#     UCB_rate = get_agent_rate_score(dirname, UCB_dirname)
#
#     trial = TrialData(N_total,
#                       N_controllable,
#                       card_A,
#                       optimal_rate,
#                       random_rate,
#                       DQN_rate,
#                       Bandit_rate,
#                       UCB_rate)
#
#     all_trial_data.append(trial)
#
#
# all_trial_data.sort(key=lambda item: item.N_controllable)
#
# df = pd.DataFrame.from_records(all_trial_data, columns=TrialData._fields)
#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# print(df.head(n=len(df)))
#
# df.to_csv('./COLLECTED_RESULTS__rates_over_N.csv')
#
#


df = pd.read_csv('./COLLECTED_RESULTS__rates_over_N.csv')


# ------------------------------------------------------

print('----------')
print('Normalized Bandit rates:')
print(df['Bandit_rate']/df['optimal_rate'])

# -----------------------------------------------------


sns.set_style('ticks')
from matplotlib import rc
rc('text', usetex=True)
fontsize = 16
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.labelsize"] = fontsize -1
plt.rcParams["ytick.labelsize"] = fontsize -1
plt.rcParams["savefig.dpi"] = 400
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.0
plt.rcParams["savefig.transparent"] = True
plt.rcParams["axes.labelsize"] = fontsize - 1
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}\usepackage{amsmath}'
#plt.rcParams['font.size'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize - 5





fig,ax = plt.subplots()

x = df['N_controllable']



CurveData = namedtuple('CurveData', ['rate_column_name',
                                     'color',
                                     'line_style',
                                     'label',
                                     'marker_style'])


colors = sns.color_palette("Set1")
line_styles = ['-','--', ':', '-.', '.']
marker_styles = ["o", "s", 'x', 'D']
hatches = ['/', '-', '\\', '|', '+', 'x', 'o', 'O', '.', '*']

all_curve_data = [
    CurveData('Bandit_rate', colors[0], '-', 'DRP',           'o'),
    CurveData('DQN_rate',    colors[1], ':', 'DQN',          'x'),
    CurveData('UCB_rate',    colors[2], '-.', 'UCB',           'D'),
    CurveData('random_rate', 'k',       '--', 'Random'        ,'*'),
    CurveData('optimal_rate', 'grey',   '--', 'Optimal'       ,'.')
]

for cd in all_curve_data:
    ax.plot(x, df[cd.rate_column_name], c=cd.color, ls=cd.line_style, marker=cd.marker_style, label=r'${\rm '+cd.label+r'}$', rasterized=False)
    #ax.scatter(x, df[cd.rate_column_name], c=cd.color, marker=cd.marker_style)




ax.set_ylabel(r"${\rm Sum\mbox{-}Rate~~(bps/Hz)}$")




plt.rcParams["xtick.minor.visible"] = False
ax.set_xlabel(r"${\rm Total~RIS~meta\mbox{-}atoms}~(N_{{\rm tot}})$")
ax.set_xticklabels([None, '$32$', '$64$', '$96$', '$128$', '$160$', None, None])




ax2 = ax.twiny()
ax2.set_xticks( ax.get_xticks() )
ax2.set_xbound(ax.get_xbound())
ax2.set_xlabel(r"${\rm Number~of~actions~}({\rm card}(\mathcal{A}))$")
ax2.set_xticklabels([None, '$16$', '$64$', '$256$', '$1024$', '$4096$', None])

ax.grid()




# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.01,
                 box.width, box.height * 1])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.2),
          fancybox=True, shadow=True, ncol=5)

plt.savefig('./plots/sum-rate-varying-N-plot.pdf')
plt.show()


#
# ind = np.arange(len(N_controllable))
# fig,ax = plt.subplots()
# width = 0.3
#
# plt.bar(ind, bandit_avg_norm, width, color=colors[0], label=r'$\text{Neural }\epsilon\text{-greedy}$', rasterized=False)
# plt.bar(ind+width, random_avg_norm, width, color=colors[1], label=r'$\text{Random policy}$', hatch=hatches[0], rasterized=False)
#
#
#
# ax.set_ylabel(r"${\rm Normalized~Sum\mbox{-}Rate}$")
# ax.set_ylim([0.3, 1.01])
#
#
# ax.set_xticks(ind + width / 2)
# ax.set_xlabel(r"${\rm Total~RIS~meta\mbox{-}atoms}~(N_{{\rm tot}})$")
# ax.set_xticklabels(['$32$', '$48$', '$64$', '$80$', '$160$'])
#
#
#
#
# ax2 = ax.twiny()
# ax2.set_xticks( ax.get_xticks() )
# ax2.set_xbound(ax.get_xbound())
# ax2.set_xlabel(r"${\rm Number~of~actions~}({\rm card}(\mathcal{A}))$")
# ax2.set_xticklabels(['$16$', '$64$', '$256$', '$1024$', '$4096$'])
#
# ax.legend()
# ax.grid()
#
# plt.savefig('./results/plots/sum-rate-varying-N-bar.pdf')
# plt.show()










