import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from tqdm import tqdm
import os, sys

from RL_experiments.environments import RISEnv2
from RL_experiments.standalone_simulatiion import Setup

sns.set_theme()


def bin2int(b):
    return b.dot(2**np.arange(b.size)[::-1])



try:
    filename = sys.argv[1]
except IndexError:
    filename = 'parameters.json'


with open(filename, 'r') as f:
    params = json.loads(f.read())



N_REALIZATIONS = 500
setup_params   = params['SETUP']
setup          = Setup(**setup_params)
env            = RISEnv2(setup, episode_length=np.inf)
num_actions    = env.action_spec().maximum + 1



rates             = np.empty((N_REALIZATIONS, num_actions))
max_rates         = np.empty(N_REALIZATIONS)
optimal_actions   = np.empty(N_REALIZATIONS, dtype=int)
optimal_profiles  = np.empty((N_REALIZATIONS, env.setup.N_controllable))
optimal_precoders = np.empty(N_REALIZATIONS)


for i in tqdm(range(N_REALIZATIONS)):

    env.reset()

    r_max         = -float('inf')
    a_best        = None
    conf_best     = None
    precoder_best = None
    (H, G, h)     = env._state

    for a in range(env.action_spec().maximum):
        codebook_selection, configuration = env.split_action_to_precoding_and_RIS_profile(a)
        configuration_ext                 = env.extend_configuration_to_groups(configuration)
        Phi                               = env.configuration2phases(configuration_ext)
        W                                 = env.select_beamforming_matrix(codebook_selection)
        #W                                 = np.ones_like(W)
        r                                 = env.compute_reward(H, G, h, Phi, W)


        rates[i,a] = r

        if r > r_max:
            r_max, a_best, conf_best = r, a, configuration

        optimal_actions[i]    = a_best
        max_rates[i]          = r_max
        optimal_profiles[i,:] = conf_best
        optimal_precoders[i]  = bin2int(codebook_selection)


fig, ax = plt.subplots(figsize=(12,6))
sns.histplot(max_rates, bins=20, ax=ax)
ax.set_title(f'Distribution of maximum rates over {N_REALIZATIONS} realizations')
plt.show()



fig, ax = plt.subplots(figsize=(12,6))
sns.histplot(optimal_actions, binwidth=1, ax=ax)
ax.set_title(f'Distribution of optimal profile indices over {N_REALIZATIONS} realizations')
plt.show()


fig, ax = plt.subplots(figsize=(12,6))
element_selection_counts = optimal_profiles.sum(axis=0)
plt.bar(np.arange(setup.N_controllable), element_selection_counts)
ax.set_title(f'Number of times each element is selected of the optimal profile in {N_REALIZATIONS} realizations')
ax.set_xlabel('RIS element')
plt.show()

# fig, ax = plt.subplots(figsize=(12,6))
# sns.histplot(optimal_precoders, binwidth=1)
# ax.set_title(f'Number of times precoding matrix is selected of the optimal profile in {N_REALIZATIONS} realizations')
# ax.set_xlabel('Precoding matrix index')
# plt.show()



