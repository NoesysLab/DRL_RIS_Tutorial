import re
import os, sys
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json





N_values= [2,4,6,8,10]

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



def fix_reward_columns(df):
    df['reward'] = df['reward'].transform(lambda x: float(x.replace("[", "").replace("]", "")))
    return df


def moving_average(x, w):
    if w % 2 != 1: w+=1

    cumsum_vec = np.cumsum(np.insert(x, 0, 0))
    return (cumsum_vec[w:] - cumsum_vec[:-w]) / w


N_controllable = 8

fname = '/home/kyriakos/workspace/reasearch/RIS/ris-codebase/RL_experiments/results/ICC/setup__N_controllable_{}_K_2_M_2_codebook_rays_per_RX_2_kappa_H_30_observation_noise_variance_0'
setup_dirname = fname.format(N_controllable)

DQN_dirnames = find_agent_dirnames('DQN', setup_dirname)
DQN_dirname = DQN_dirnames[0]

Bandit_dirnames = find_agent_dirnames('Neural Softmax', setup_dirname)
Bandit_dirname = Bandit_dirnames[0]







df_DQN = pd.read_csv(setup_dirname+'/'+DQN_dirname+"/agent_training.csv")
df_DQN = fix_reward_columns(df_DQN)



df_Bandit = pd.read_csv(setup_dirname+'/'+Bandit_dirname+"/agent_training.csv")

df = df_DQN.merge(df_Bandit, left_on='iteration', right_on='iteration', suffixes=('_DQN', '_Bandit'))


print(df.head())



# --------------------------------------------


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
plt.rcParams['legend.fontsize'] = fontsize - 1


colors = sns.color_palette("Set1")
line_styles = ['-','--', ':', '-.', '.']
marker_styles = ["o", "s", 'x', 'D']
hatches = ['/', '-', '\\', '|', '+', 'x', 'o', 'O', '.', '*']

W = 300

FIRST_N = 15000
df = df[df['iteration']<=FIRST_N]

print(df['iteration'].max())

DQN_rewards_smoothed    = df['reward_DQN'].rolling(window=W).mean().values
DQN_rewards_std         = df['reward_DQN'].rolling(window=W).std().values
Bandit_rewards_smoothed = df['reward_Bandit'].rolling(window=W).mean().values
Bandit_rewards_std      = df['reward_Bandit'].rolling(window=W).std().values

assert len(DQN_rewards_smoothed) == len(Bandit_rewards_smoothed)

x = range(1, len(Bandit_rewards_smoothed)+1)



fig, ax = plt.subplots()

ax.plot(x, Bandit_rewards_smoothed, color=colors[0], label=r'${\rm DRP}$')
ax.plot(x, DQN_rewards_smoothed, color=colors[1], label=r'${\rm DQN}$')

ax.fill_between(x, Bandit_rewards_smoothed-Bandit_rewards_std, Bandit_rewards_smoothed+Bandit_rewards_std, color=colors[0], alpha=0.5)
ax.fill_between(x, DQN_rewards_smoothed-Bandit_rewards_std, DQN_rewards_smoothed+Bandit_rewards_std, color=colors[1], alpha=0.5)

plt.legend(loc='lower right')
plt.grid()
ax.set_xlabel(r'${\rm Iteration}$')
ax.set_ylabel(r"${\rm Sum\mbox{-}Rate~~(bps/Hz)}$")
ax.set_xlim([1,FIRST_N ])
plt.savefig("./plots/convergence-rate.pdf")
plt.show()



print(Bandit_rewards_smoothed-Bandit_rewards_std)


