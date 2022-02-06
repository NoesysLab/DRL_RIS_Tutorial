import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys, os

results_rootdir = "../results/Moving_UEs/"



#
# Setup Parameters
#
N_controllable = 2
kappa_H        = 15



#
# Experiment Parameters
#
n_iters        = 50
n_evals        = 15

#
# Methoods
#
DQN_dir           = 'DQN_learning_rate_0.0002_target_update_tau_0.18'
NeuralSoftmax_dir = 'Neural Softmax_learning_rate_0.002_Boltzmann_tau_0.6'
UCB_dir           = 'UCB_alpha_0.6_gamma_0.75'




####################################################################
setup_dir = f"Setup_N_contr_{N_controllable}_kappa_H_{kappa_H}/"


Exhaustive_results_fname = results_rootdir + setup_dir + "Baselines/" + f"n_iters_{n_iters}_n_evals_{n_evals}/Exhaustive/evaluation_performance.csv"
Random_results_fname    = results_rootdir + setup_dir + "Baselines/" + f"n_iters_{n_iters}_n_evals_{n_evals}/Random/evaluation_performance.csv"

UCB_results_fname_channels = results_rootdir + setup_dir + "Agents/" + f"channels_n_iters_{n_iters}_n_evals_{n_evals}/" + UCB_dir + "/performance.csv"
UCB_results_fname_angles   = results_rootdir + setup_dir + "Agents/" + f"angles_n_iters_{n_iters}_n_evals_{n_evals}/" + UCB_dir + "/performance.csv"

DQN_results_fname_channels = results_rootdir + setup_dir + "Agents/" + f"channels_n_iters_{n_iters}_n_evals_{n_evals}/" + DQN_dir + "/performance.csv"
DQN_results_fname_angles   = results_rootdir + setup_dir + "Agents/" + f"angles_n_iters_{n_iters}_n_evals_{n_evals}/" + DQN_dir + "/performance.csv"

NeuralSoftmax_results_fname_channels = results_rootdir + setup_dir + "Agents/" + f"channels_n_iters_{n_iters}_n_evals_{n_evals}/" + NeuralSoftmax_dir + "/performance.csv"
NeuralSoftmax_results_fname_angles   = results_rootdir + setup_dir + "Agents/" + f"angles_n_iters_{n_iters}_n_evals_{n_evals}/" + NeuralSoftmax_dir + "/performance.csv"



if __name__ == '__main__':

    title = f"N_contr_{N_controllable}_kappa_H_{kappa_H}"

    df_exhaustive             = pd.read_csv(Exhaustive_results_fname)
    df_random                 = pd.read_csv(Random_results_fname)
    df_UCB_channels           = pd.read_csv(UCB_results_fname_channels)
    df_UCB_angles             = pd.read_csv(UCB_results_fname_angles)
    df_DQN_channels           = pd.read_csv(DQN_results_fname_channels)
    df_DQN_angles             = pd.read_csv(DQN_results_fname_angles)
    df_NeuralSoftmax_channels = pd.read_csv(NeuralSoftmax_results_fname_channels)
    df_NeuralSoftmax_angles   = pd.read_csv(NeuralSoftmax_results_fname_angles)


    reward_col = 'rewards_wrt_optimal'
    steps = df_exhaustive['Time Step']

    random_norm_rewards = df_random['Reward'] / df_exhaustive['Reward']

    fig, ax = plt.subplots()

    #plt.plot(steps, df_exhaustive['Reward'], c='k', label='Exhaustive')
    plt.plot(steps, random_norm_rewards, c='k', label='random')

    plt.plot(steps, df_UCB_channels[reward_col], 'r+-', label='UCB (channels)')
    plt.plot(steps, df_UCB_angles[reward_col],  'r+-.', label='UCB (AoDs)')

    plt.plot(steps, df_DQN_channels[reward_col], 'bo-', label='DQN (channels)')
    plt.plot(steps, df_DQN_angles[reward_col], 'bo-.', label='DQN (AoDs)')

    plt.plot(steps, df_NeuralSoftmax_channels[reward_col], 'gd-', label='NeuralSoftmax (channels)')
    plt.plot(steps, df_NeuralSoftmax_angles[reward_col], 'gd-.', label='NeuralSoftmax (AoDs)')

    plt.title(title)
    plt.legend()
    plt.show()


    labels = ['DQN', 'NeuralSoftmax', 'UCB',]
    channels_avg_rewards = [df_DQN_channels[reward_col].mean(),
                            df_NeuralSoftmax_channels[reward_col].mean(),
                            df_UCB_channels[reward_col].mean(),
                            ]
    angles_avg_rewards   = [df_DQN_angles[reward_col].mean(),
                            df_NeuralSoftmax_angles[reward_col].mean(),
                            df_UCB_angles[reward_col].mean(),]



    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, channels_avg_rewards, width, label='Channels')
    rects2 = ax.bar(x + width / 2, angles_avg_rewards, width, label='AoDs')

    ax.hlines(random_norm_rewards.mean(), 0-width, x[-1]+width, colors=[(0,0,0)], linestyles='dashed', label='Random baseline')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.title(title)
    plt.show()


























