import re
import os, sys
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


#
# Setup Parameters
#
N_controllable_vals = [2,4,6,8,10]
kappa_H             = 30


def load_data(N_contr, kappa_H):

    results_rootdir = "../all_setup_parameters/results/Rates/"

    #
    # Experiment Parameters
    #
    n_iters        = 50
    n_evals        = 15

    #
    # Methoods
    #
    DQN_dir           = 'DQN_learning_rate_0.0002_target_update_tau_0.18'
    NeuralSoftmax_dir = 'Neural Softmax_learning_rate_0.001_Boltzmann_tau_0.6'
    UCB_dir           = 'UCB_alpha_0.6_gamma_0.75'

    if N_contr == 2:
        NeuralSoftmax_dir = 'Neural Softmax_learning_rate_0.002_Boltzmann_tau_0.6'




    ####################################################################
    setup_dir_tpl = "Setup_N_contr_{N_controllable}_kappa_H_{kappa_H}/"
    setup_dir = setup_dir_tpl.format(N_controllable=N_contr, kappa_H=kappa_H)


    Exhaustive_results_fname = results_rootdir + setup_dir + "Baselines/" + f"n_iters_{n_iters}_n_evals_{n_evals}/Exhaustive/evaluation_performance.json"
    Random_results_fname    = results_rootdir + setup_dir + "Baselines/" + f"n_iters_{n_iters}_n_evals_{n_evals}/Random/evaluation_performance.json"

    UCB_results_fname = results_rootdir + setup_dir + "Agents/" + f"channels_n_iters_{n_iters}_n_evals_{n_evals}/" + UCB_dir + "/performance.csv"

    DQN_results_fname = results_rootdir + setup_dir + "Agents/" + f"channels_n_iters_{n_iters}_n_evals_{n_evals}/" + DQN_dir + "/performance.csv"

    NeuralSoftmax_results_fname = results_rootdir + setup_dir + "Agents/" + f"channels_n_iters_{n_iters}_n_evals_{n_evals}/" + NeuralSoftmax_dir + "/performance.csv"


    Exhaustive_rate = json.load(open(Exhaustive_results_fname))['Mean Reward']
    Random_rate     = json.load(open(Random_results_fname))['Mean Reward']
    UCB_rate        = pd.read_csv(open(UCB_results_fname))['Reward'].values.max()
    DQN_rate        = pd.read_csv(open(DQN_results_fname))['Reward'].values.max()
    NSM_rate        = pd.read_csv(open(NeuralSoftmax_results_fname))['Reward'].values.max()

    return Exhaustive_rate, Random_rate, UCB_rate, DQN_rate, NSM_rate





def plot2():

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

    x = N_controllable_vals



    CurveData = namedtuple('CurveData', ['rates',
                                         'color',
                                         'line_style',
                                         'label',
                                         'hatch'])


    colors = sns.color_palette("Set1")
    line_styles = ['-','--', ':', '-.', '.']
    marker_styles = ["o", "s", 'x', 'D']
    hatches = ['/', '-', '\\', '|', '+', 'x', 'o', 'O', '.', '*']

    all_curve_data = [
        CurveData(NSM_rates/Exhaustive_rates, colors[0], '-', r'{\rm Neural}$ \ \epsilon${\rm -greedy}',           '/'),
        CurveData(DQN_rates/Exhaustive_rates,    colors[1], ':', r'{\rm DQN}',          '\\'),
        CurveData(UCB_rates/Exhaustive_rates,    colors[2], '-.', r'{\rm UCB}',           '-'),
        CurveData(Random_rates/Exhaustive_rates, 'grey',       '--', r'{\rm Random Policy}'        ,None),
#        CurveData(Exhaustive_rates, 'k',   '--', 'Optimal'       ,'.')
    ]




    ax.set_ylabel(r"${\rm Sum\mbox{-}Rate~~(bps/Hz)}$")

    x = np.arange(len(N_controllable_vals))  # the label locations
    width = 0.8  # the width of the bars

    fig, ax = plt.subplots()

    rects = []

    for cd, offset in zip(all_curve_data, [-width/2, -width/4, 0, width/4]):
        rects_i = ax.bar(x + offset, cd.rates, width/4,
                        label=cd.label,
                        color=cd.color,
                        hatch=cd.hatch,
                         align='edge',
                        )

        rects.append(rects_i)


    # x_lim_min = 0 - width - width / 4
    # x_lim_max = x[-1] + width / 2 + width / 4

    # ax.hlines(random_norm_rewards.mean(), x_lim_min, x_lim_max,
    #           colors=[(0, 0, 0)], linestyles='dashed', label=r'{\rm Random policy}')

    ax.set_ylabel(r'{\rm Normalized Sum-Rate}')
    #ax.set_xticks(np.arange(len(all_curve_data) - 2))
    #x.set_xticklabels([cd.label for cd in list(all_curve_data.values())[:-2]])

    #ax.set_xlim([x_lim_min, x_lim_max])

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [1, 2, 3, 0]
    # ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title=r'{\rm Observation Type}',
    #           loc='lower right')





    plt.rcParams["xtick.minor.visible"] = False
    ax.set_xlabel(r"${\rm Total~RIS~meta\mbox{-}atoms}~(N_{{\rm tot}})$")
    ax.set_xticklabels([None, '$32$', '$64$', '$96$', '$128$', '$160$', None,])




    ax2 = ax.twiny()
    ax2.set_xticks( ax.get_xticks() )
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xlabel(r"${\rm Number~of~actions~}({\rm card}(\mathcal{A}))$")
    ax2.set_xticklabels([None, '$16$', '$64$', '$256$', '$1024$', '$4096$', None])

    ax.grid()




    # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.01,
    #                  box.width, box.height * 1])

    # Put a legend below current axis
    ax.legend(#loc='upper center',
              #bbox_to_anchor=(0.45, -0.2),
              loc='upper right',
              fancybox=True, shadow=True, ncol=2)

    plt.savefig('./plots/sum-rate-varying-N-plot-normalized.pdf')
    plt.savefig('./plots/sum-rate-varying-N-plot-normalized.png')
    plt.show()




if __name__ == "__main__":
    Exhaustive_rates = np.empty(len(N_controllable_vals))
    Random_rates = np.empty_like(Exhaustive_rates)
    UCB_rates = np.empty_like(Exhaustive_rates)
    DQN_rates = np.empty_like(Exhaustive_rates)
    NSM_rates = np.empty_like(Exhaustive_rates)


    for i, N_contr in enumerate(N_controllable_vals):
        Exhaustive_rates[i], Random_rates[i], UCB_rates[i], DQN_rates[i], NSM_rates[i] = load_data(N_contr, kappa_H)

    plot2()




