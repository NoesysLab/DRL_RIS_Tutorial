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
N_controllable      = 6
kappa_H             = 30
P_vals              = [10, 20, 30 , 40, 50]

def load_data(P):

    results_rootdir = "../all_setup_parameters/results/Power/"

    #
    # Experiment Parameters
    #
    n_iters        = 50
    n_evals        = 5

    #
    # Methoods
    #
    DQN_dir           = 'DQN_learning_rate_0.0002_target_update_tau_0.18'
    NeuralSoftmax_dir = 'Neural Softmax_learning_rate_0.001_Boltzmann_tau_0.6'
    UCB_dir           = 'UCB_alpha_0.6_gamma_0.75'





    ####################################################################
    setup_dir_tpl = "Setup_N_contr_{N_controllable}_kappa_H_{kappa_H}_P_{P}/"
    setup_dir = setup_dir_tpl.format(N_controllable=N_controllable, kappa_H=kappa_H, P=P)


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





def plot1():

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
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["axes.labelsize"] = fontsize - 1
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}\usepackage{amsmath}'
    #plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize - 5





    fig,ax = plt.subplots()

    x = P_vals



    CurveData = namedtuple('CurveData', ['rates',
                                         'color',
                                         'line_style',
                                         'label',
                                         'marker_style'])


    colors = sns.color_palette("Set1")
    line_styles = ['-','--', ':', '-.', '.']
    marker_styles = ["o", "s", 'x', 'D']
    hatches = ['/', '-', '\\', '|', '+', 'x', 'o', 'O', '.', '*']

    all_curve_data = [
        CurveData(NSM_rates, colors[0], '-', r'Neural~}\epsilon{\rm -greedy',           'o'),
        CurveData(DQN_rates,    colors[1], ':', 'DQN',          'x'),
        CurveData(UCB_rates,    colors[2], '-.', 'UCB',           'D'),
        CurveData(Random_rates, 'k',       '--', 'Random'        ,'*'),
        CurveData(Exhaustive_rates, 'grey',   '--', 'Optimal'       ,'.')
    ]

    for cd in all_curve_data:
        ax.plot(x, cd.rates, c=cd.color, ls=cd.line_style, label=r'${\rm '+cd.label+r'}$', rasterized=False, marker=cd.marker_style)




    ax.set_ylabel(r"${\rm Sum~rate~(bps/Hz)}$")




    plt.rcParams["xtick.minor.visible"] = False
    ax.set_xlabel(r"${\rm Transmit~power~}(P){\rm ~in~ dBm}$")
    #ax.set_xticklabels([None, '$10$', '$10$', '$30$', '$40$', '$50$', None, None])


    ax.grid()


    #ax.set_yscale('log')

    #inset axes....
    start_x = 12
    end_x = 35
    start_y = 2
    end_y   = 4.8
    axins = ax.inset_axes([start_x, start_y, end_x-start_x, end_y-start_y], transform=ax.transData, alpha=0.5)

    for cd in all_curve_data:
        axins.plot(x[:2], cd.rates[:2], c=cd.color, ls=cd.line_style, marker=cd.marker_style)

    axins.grid()
    #axins.set_yscale('log')



    # sub region of the original image
    x1, x2, y1, y2 = 15, 35, 2.5, 4.5

    #axins.set_xlim(x1, x2)
    #axins.set_ylim(y1, y2)

    ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.legend(loc='lower right', ncol=3)
    ax.set_ylim([-0.999, 5])

    plt.savefig('./plots/sum-rate-varying-P-plot-inset.pdf')
    plt.savefig('./plots/sum-rate-varying-P-plot-inset.png')

    plt.show()




def plot2():
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
    plt.rcParams['legend.fontsize'] = fontsize - 5

    fig, ax = plt.subplots()

    x = P_vals

    CurveData = namedtuple('CurveData', ['rates',
                                         'color',
                                         'line_style',
                                         'label',
                                         'hatch'])

    colors = sns.color_palette("Set1")
    line_styles = ['-', '--', ':', '-.', '.']
    marker_styles = ["o", "s", 'x', 'D']
    hatches = ['/', '-', '\\', '|', '+', 'x', 'o', 'O', '.', '*']

    all_curve_data = [
        CurveData(NSM_rates / Exhaustive_rates, colors[0], '-', r'{\rm Neural}$ \ \epsilon${\rm -greedy}', '/'),
        CurveData(DQN_rates / Exhaustive_rates, colors[1], ':', r'{\rm DQN}', '\\'),
        CurveData(UCB_rates / Exhaustive_rates, colors[2], '-.', r'{\rm UCB}', '-'),
        CurveData(Random_rates / Exhaustive_rates, 'grey', '--', r'{\rm Random}', None),
        #        CurveData(Exhaustive_rates, 'k',   '--', 'Optimal'       ,'.')
    ]

    ax.set_ylabel(r"${\rm Sum\mbox{-}Rate~~(bps/Hz)}$")

    x = np.arange(len(P_vals))  # the label locations
    width = 0.8  # the width of the bars

    fig, ax = plt.subplots()

    rects = []

    for cd, offset in zip(all_curve_data, [-width / 2, -width / 4, 0, width / 4]):
        rects_i = ax.bar(x + offset, cd.rates, width / 4,
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

    ax.set_ylabel(r'{\rm Normalized sum rate}')
    # ax.set_xticks(np.arange(len(all_curve_data) - 2))
    # x.set_xticklabels([cd.label for cd in list(all_curve_data.values())[:-2]])

    # ax.set_xlim([x_lim_min, x_lim_max])

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [1, 2, 3, 0]
    # ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title=r'{\rm Observation Type}',
    #           loc='lower right')

    # plt.rcParams["xtick.minor.visible"] = False
    # ax.set_xlabel(r"${\rm Total~RIS~unit\mbox{-}elements}~(N_{{\rm tot}})$")
    # ax.set_xticklabels([None, '$32$', '$64$', '$96$', '$128$', '$160$', None, ])
    #
    # ax2 = ax.twiny()
    # ax2.set_xticks(ax.get_xticks())
    # ax2.set_xbound(ax.get_xbound())
    # ax2.set_xlabel(r"${\rm Number~of~actions~}({\rm card}(\mathcal{A}))$", labelpad=10)
    # ax2.set_xticklabels([None, '$16$', '$64$', '$256$', '$1024$', '$4096$', None])

    ax.set_xlabel(r"${\rm Transmit~power~}(P){\rm ~in~ dBm}$")
    ax.set_xticklabels([None, '$10$', '$20$', '$30$', '$40$', '$50$'])


    ax.grid()

    # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.01,
    #                  box.width, box.height * 1])


    ax.set_ylim([None, 1.])

    # Put a legend below current axis
    ax.legend(  # loc='upper center',
        # bbox_to_anchor=(0.45, -0.2),
        loc='upper right',
        fancybox=True, shadow=True, ncol=2)

    plt.savefig('./plots/sum-rate-varying-P-plot-normalized.pdf')
    plt.savefig('./plots/sum-rate-varying-P-plot-normalized.png')
    plt.show()


if __name__ == "__main__":
    Exhaustive_rates = np.empty(len(P_vals))
    Random_rates = np.empty_like(Exhaustive_rates)
    UCB_rates = np.empty_like(Exhaustive_rates)
    DQN_rates = np.empty_like(Exhaustive_rates)
    NSM_rates = np.empty_like(Exhaustive_rates)


    for i, P in enumerate(P_vals):
        Exhaustive_rates[i], Random_rates[i], UCB_rates[i], DQN_rates[i], NSM_rates[i] = load_data(P)

    plot1()
    plot2()




