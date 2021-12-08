import re
import os, sys
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

results_path = './'


def get_top_level_dirnames(path):
    return next(os.walk(path))[1]


def find_agent_dirnames(agent_name, setup_dirname):
    all_agent_dirnames  = get_top_level_dirnames(setup_dirname)
    this_agent_dirnames = []

    for this_agent_dirname in all_agent_dirnames:
        re_agent = re.compile(agent_name)
        m = re.search(re_agent, this_agent_dirname)
        if m:
            this_agent_dirnames.append(this_agent_dirname)

    if len(this_agent_dirnames) == 0:
        print(f" > Not found any directories for {agent_name}")
    else:
        print(f" > Found directories for {agent_name}:")
        for dirname in this_agent_dirnames:
            print(f"  -  {dirname}")

    return this_agent_dirnames


def get_agent_rate_score(setup_dirname, agent_dirname):
    with open(setup_dirname + "/" + agent_dirname + "/agent_performance.json") as fagent:
        agent_performance = json.load(fagent)
        rate = float(agent_performance['avg_score'])

        return rate


re_setup_dirname = re.compile(r'setup__transmit_power_\d+_N_controllable_6')





TrialData = namedtuple('TrialData', ['Power',
                                     'N_total',
                                     'N_controllable',
                                     'card_A',
                                     'optimal_rate',
                                     'random_rate',
                                     'DQN_rate',
                                     'Bandit_rate',])


all_trial_data = []


for dirname in get_top_level_dirnames(results_path):

    m = re.match(re_setup_dirname, dirname)

    if m is None:
        print(f"Not matched: {dirname}")
        continue

    print(f'Processing trial with dirname "{dirname}"')



    with open(dirname+'/setup.json') as fsetup:
        setup_params   = json.load(fsetup)
        Power          = int(int(setup_params['transmit_power']))
        N_total        = int(setup_params['N']) * int(setup_params['M'])
        N_controllable = int(setup_params['N_controllable'])
        card_A         = 2 ** int(N_controllable + 2)

    with open(dirname+'/baselines.json') as fbaselines:
        baselines_scores = json.load(fbaselines)
        optimal_rate     = float(baselines_scores['optimal_score'])
        random_rate      = float(baselines_scores['random_policy_average_return'])


    DQN_dirnames = find_agent_dirnames('DQN', dirname)
    DQN_dirname  = DQN_dirnames[0]
    DQN_rate     = get_agent_rate_score(dirname, DQN_dirname)


    Bandit_dirnames = find_agent_dirnames('Neural Softmax', dirname)
    Bandit_dirname  = Bandit_dirnames[0]
    Bandit_rate     = get_agent_rate_score(dirname, Bandit_dirname)

    # UCB_dirnames    = find_agent_dirnames('UCB', dirname)
    # UCB_dirname     = None
    # re_UCB_dirname  = re.compile("alpha_10")
    # for UCB_dir in UCB_dirnames:
    #     m_ = re.search(re_UCB_dirname, UCB_dir)
    #     if m_:
    #         print(f' + Using "{UCB_dir}" for UCB')
    #         UCB_dirname = UCB_dir
    #         break
    # UCB_rate = get_agent_rate_score(dirname, UCB_dirname)

    trial = TrialData(Power,
                      N_total,
                      N_controllable,
                      card_A,
                      optimal_rate,
                      random_rate,
                      DQN_rate,
                      Bandit_rate,)

    all_trial_data.append(trial)


all_trial_data.sort(key=lambda item: item.Power)

df = pd.DataFrame.from_records(all_trial_data, columns=TrialData._fields)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.head(n=len(df)))

df.to_csv('./COLLECTED_RESULTS__rates_over_power.csv')




# df['optimal_rate'] = df['optimal_rate'].values[0]
# df['random_rate'] = df['random_rate'].values[0]


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

x = df['Power']
print(x)


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
    # CurveData('UCB_rate',    colors[2], '-.', 'UCB',           'D'),
    # CurveData('random_rate', 'k',       '--', 'Random'        ,'*'),
    # CurveData('optimal_rate', 'grey',   '--', 'Optimal'       ,'.')
]

for cd in all_curve_data:
    ax.plot(x, df[cd.rate_column_name], c=cd.color, ls=cd.line_style, label=r'${\rm '+cd.label+r'}$', rasterized=False)
    ax.scatter(x, df[cd.rate_column_name], c=cd.color, marker=cd.marker_style)

ax.plot(x, df['optimal_rate'], c='grey', ls='--', label=r'${\rm Optimal}$', rasterized=False)
ax.plot(x, df['random_rate'], c='k', ls='--', label=r'${\rm Random}$', rasterized=False)





ax.set_ylabel(r"${\rm Sum\mbox{-}Rate~~(bps/Hz)}$")




plt.rcParams["xtick.minor.visible"] = False
ax.set_xlabel(r"${\rm Transmit Power~(P)}$")
ax.set_xticklabels([None, '$10$', '$20$', '$30$', '$40$', '$50$', None])




ax.grid()




# # Shrink current axis's height by 10% on the bottom
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.01,
#                  box.width, box.height * 1])
#
# # Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.2),
#           fancybox=True, shadow=True, ncol=5)

plt.legend()

plt.savefig('./plots/rate-varying-power-plot.pdf')
plt.show()



# -----------------------------------------------------------------------

Bandit_norm_rates = (df['Bandit_rate']/df['optimal_rate']).values
DQN_norm_rates = (df['DQN_rate']/df['optimal_rate']).values

# Bandit_norm_rates = df['Bandit_rate'].values
# DQN_norm_rates = df['DQN_rate'].values


print('\t\t\t' + r'$P~{\rm (dBm)}$ & ' + " & ".join(map(str,[10,20,30,40,50]))+" \\")
print('\t\t\t' + r'\hline')
print('\t\t\t' + 'DRP  & ' + " & ".join([f"{r:.3f}" for r in Bandit_norm_rates])+" \\")
print('\t\t\t' + 'DQN &'  + " & ".join([f"{r:.3f}" for r in DQN_norm_rates])+" \\")
print('\t\t\t' + r'\hline')









