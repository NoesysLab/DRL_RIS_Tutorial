from collections import namedtuple, OrderedDict

import cycler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys, os





def load_data(N_contr):

    results_rootdir = "../results/Moving_UEs/"



    #
    # Setup Parameters
    #
    N_controllable = N_contr
    kappa_H        = 30



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

    df_exhaustive             = pd.read_csv(Exhaustive_results_fname)
    df_random                 = pd.read_csv(Random_results_fname)
    df_UCB_channels           = pd.read_csv(UCB_results_fname_channels)
    df_UCB_angles             = pd.read_csv(UCB_results_fname_angles)
    df_DQN_channels           = pd.read_csv(DQN_results_fname_channels)
    df_DQN_angles             = pd.read_csv(DQN_results_fname_angles)
    df_NeuralSoftmax_channels = pd.read_csv(NeuralSoftmax_results_fname_channels)
    df_NeuralSoftmax_angles   = pd.read_csv(NeuralSoftmax_results_fname_angles)

    return df_exhaustive, df_random, df_UCB_channels, df_UCB_angles, df_DQN_channels, df_DQN_angles, df_NeuralSoftmax_channels, df_NeuralSoftmax_angles




def multi_bar_plot(N_contr, df_exhaustive, df_random, df_UCB_channels, df_UCB_angles, df_DQN_channels, df_DQN_angles, df_NeuralSoftmax_channels, df_NeuralSoftmax_angles):



    sns.set_style('ticks')
    from matplotlib import rc

    rc('text', usetex=True)
    fontsize = 16
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["xtick.labelsize"] = fontsize - 1
    plt.rcParams["ytick.labelsize"] = fontsize - 1
    plt.rcParams["savefig.dpi"] = 400
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.0
    plt.rcParams["savefig.transparent"] = True
    plt.rcParams["axes.labelsize"] = fontsize - 1
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}\usepackage{amsmath}'
    # plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize - 3
    plt.rcParams['legend.title_fontsize'] = fontsize - 2


    CurveData = namedtuple('CurveData', [ 'color',
                                         'line_style',
                                         'label',
                                         'marker_style'])

    colors = sns.color_palette("Set1")
    line_styles = ['-', '--', ':', '-.', '.']
    marker_styles = ["o", "s", 'x', 'D']
    hatches = ['/', '-', '\\', '|', '+', 'x', 'o', 'O', '.', '*']

    all_curve_data = OrderedDict( {
        'DQN': CurveData(colors[1], ':', r'{\rm DQN}', 'x'),
        'DRP': CurveData(colors[0], '-', r'{\rm Neural}$\ \epsilon${\rm-greedy}', 'o'),
        'UCB': CurveData(colors[2], '-.', r'{\rm UCB}', 'D'),
        'Random': CurveData('k', '--', r'{\rm Random policy}', '*'),
        'Exhaustive': CurveData('grey', '--', r'{\rm Optimal Policy}', '.')
    } )





    reward_col = 'rewards_wrt_optimal'
    steps = df_exhaustive['Time Step']

    random_norm_rewards = df_random['Reward'] / df_exhaustive['Reward']

    # fig, ax = plt.subplots()
    #
    # #plt.plot(steps, df_exhaustive['Reward'], c='k', label='Exhaustive')
    # plt.plot(steps, random_norm_rewards, c='k', label='random')
    #
    # plt.plot(steps, df_UCB_channels[reward_col], 'r+-', label='UCB (channels)')
    # plt.plot(steps, df_UCB_angles[reward_col],  'r+-.', label='UCB (AoDs)')
    #
    # plt.plot(steps, df_DQN_channels[reward_col], 'bo-', label='DQN (channels)')
    # plt.plot(steps, df_DQN_angles[reward_col], 'bo-.', label='DQN (AoDs)')
    #
    # plt.plot(steps, df_NeuralSoftmax_channels[reward_col], 'gd-', label='NeuralSoftmax (channels)')
    # plt.plot(steps, df_NeuralSoftmax_angles[reward_col], 'gd-.', label='NeuralSoftmax (AoDs)')
    #
    # plt.title(title)
    # plt.legend()
    # plt.show()


    channels_avg_rewards = [df_DQN_channels[reward_col].mean(),
                            df_NeuralSoftmax_channels[reward_col].mean(),
                            #df_UCB_channels[reward_col].mean(),
                            ]
    angles_avg_rewards   = [df_DQN_angles[reward_col].mean(),
                            df_NeuralSoftmax_angles[reward_col].mean(),
                            #df_UCB_angles[reward_col].mean(),
                            ]

    channel_std_rewards  = [df_DQN_channels[reward_col].std(),
                            df_NeuralSoftmax_channels[reward_col].std(),
                            #df_UCB_channels[reward_col].std(),
                            ]
    angles_std_rewards   = [df_DQN_angles[reward_col].std(),
                            df_NeuralSoftmax_angles[reward_col].std(),
                            #df_UCB_angles[reward_col].std(),
                            ]

    hatch_channels   = None
    hatch_angles     = '\\'
    hatch_no_obs     = '|'


    import wes
    cccs = cycler.cycler(color=wes._cycles['GrandBudapest2'])
    cccs = [list(d.values())[0] for d in list(cccs)]
    obs_type_colors = cccs#sns.color_palette('Paired')
    color_channels = obs_type_colors[0]
    color_angles   = obs_type_colors[1]
    color_no_obs   = all_curve_data['UCB'].color


    x = np.arange(len(all_curve_data)-2)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x[:-1] - width / 2, channels_avg_rewards, width, label=r'{\rm Channels}', yerr=channel_std_rewards,
                    color=color_channels, hatch=hatch_channels,
                    #color=[all_curve_data['DQN'].color, all_curve_data['DRP'].color], hatch=hatch_channels

                    )
    rects2 = ax.bar(x[:-1] + width / 2, angles_avg_rewards, width, label=r'{\rm AoDs}',yerr=angles_std_rewards,
                    color=color_angles, hatch=hatch_angles,
                    #color=[all_curve_data['DQN'].color, all_curve_data['DRP'].color], hatch=hatch_angles,
                    )

    rects3 = ax.bar(x[-1], df_UCB_channels[reward_col].mean(), width, yerr=df_UCB_channels[reward_col].std(),
                    label=r'{\rm No observation}',
                    color=color_no_obs, hatch=hatch_no_obs,
                    #label='No observation', color=all_curve_data['UCB'].color, hatch=hatch_no_obs,
                    )


    x_lim_min = 0 - width - width/4
    x_lim_max = x[-1] + width/2 + width/4

    ax.hlines(random_norm_rewards.mean(), x_lim_min, x_lim_max,
              colors=[(0, 0, 0)], linestyles='dashed', label=r'{\rm Random policy}')

    ax.set_ylabel(r'{\rm Normalized Sum-Rate}')
    ax.set_xticks(np.arange(len(all_curve_data)-2))
    ax.set_xticklabels([cd.label for cd in  list(all_curve_data.values())[:-2]])
    ax.set_xlim([x_lim_min, x_lim_max])

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,3,0]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title=r'{\rm Observation Type}', loc='lower right')

    plt.grid()

    fig.tight_layout()
    plt.savefig(f'./plots/channels_vs_angles_barplots_N_contr{N_contr}.png')
    plt.savefig(f'./plots/channels_vs_angles_barplots_N_contr{N_contr}.pdf')
    plt.show(block=False)


if __name__ == '__main__':
    N_contr = 4
    df_exhaustive, df_random, df_UCB_channels, df_UCB_angles, df_DQN_channels, df_DQN_angles, df_NeuralSoftmax_channels, df_NeuralSoftmax_angles = load_data(N_contr)
    multi_bar_plot(N_contr, df_exhaustive, df_random, df_UCB_channels, df_UCB_angles, df_DQN_channels, df_DQN_angles, df_NeuralSoftmax_channels, df_NeuralSoftmax_angles)

    N_contr = 8
    df_exhaustive, df_random, df_UCB_channels, df_UCB_angles, df_DQN_channels, df_DQN_angles, df_NeuralSoftmax_channels, df_NeuralSoftmax_angles = load_data(N_contr)
    multi_bar_plot(N_contr, df_exhaustive, df_random, df_UCB_channels, df_UCB_angles, df_DQN_channels, df_DQN_angles, df_NeuralSoftmax_channels, df_NeuralSoftmax_angles)























