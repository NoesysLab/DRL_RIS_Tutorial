import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from core.simulation import Simulator



def plot_correlation_of_elements(Phases, title_extra=''):
    df = pd.DataFrame(data=Phases)
    corr_mat = df.corr().stack().reset_index(name="correlation")
    g = sns.relplot(
        data=corr_mat,
        x="level_0", y="level_1", hue="correlation", size="correlation",
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
        height=10, sizes=(50, 250), size_norm=(-.2, .8),
    )
    # Tweak the figure to finalize
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    #g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)

    if g.legend is not None:
        for artist in g.legend.legendHandles:
            artist.set_edgecolor(".7")

    if title_extra:
        plt.title(title_extra)



def plot_phase_elements_distributions(Phases, phase_names=None, title_extra=''):
    if phase_names is None:
        phase_names = ['ones', 'zeros']

    df = pd.DataFrame(data=Phases)
    ones  = df.sum() / float(len(df))
    zeros = ones.copy()
    zeros[:] = 1

    colors = sns.color_palette("Paired")
    c1 = colors[8]
    c2 = colors[9]

    sns.barplot(x=zeros.index, y=zeros.values, label=phase_names[0], color=c1)
    sns.barplot(x=ones.index, y=ones.values, label=phase_names[1], color=c2)

    plt.title("Distribution of phases "+title_extra)
    plt.xlabel("RIS element")
    plt.ylabel(f"Relative frequency ({len(df)} trials)")
    plt.hlines(.5, -.5, len(df.columns)-.5, colors='k', linestyles='--')
    plt.legend(title='phase values')


def plot_snr_distribution(SNRs, title_extra=''):
    sns.displot(SNRs)
    plt.ylabel(f'Counts ({len(SNRs)} trials)')
    plt.title('Optimal SNR values '+title_extra, size=9)



def plot_channels(Hs,
                  Gs,
                  h0s,
                  num_ris):



    def plot_complex_vector_statistics(V_real_or_imag_mean,
                                       V_real_or_imag_std,
                                       ax,
                                       title):
        x = range(len(V_real_or_imag_mean))
        ax.scatter     (x, V_real_or_imag_mean)
        ax.plot        (x, V_real_or_imag_mean, alpha=0.5)
        ax.fill_between(x,
                           V_real_or_imag_mean - V_real_or_imag_std,
                           V_real_or_imag_mean + V_real_or_imag_std,
                           alpha=0.25)
        ax.set_xlabel('RIS element')
        ax.set_title(title)
        for i in range(0, len(V_real_or_imag_mean), len(V_real_or_imag_mean) // num_ris):
            ax.axvline(i - 0.5, c='k', ls=':')


    fig, ax = plt.subplots(2,1)

    H_means_real = Hs.real.mean(axis=0)
    H_means_imag = Hs.imag.mean(axis=0)
    H_stds_real  = Hs.real.std(axis=0)
    H_stds_imag  = Hs.imag.std(axis=0)

    plot_complex_vector_statistics(H_means_real, H_stds_real, ax[0], 'Re{H}')
    plot_complex_vector_statistics(H_means_imag, H_stds_imag, ax[1], 'Im{H}')

def mean_stds_phases(Phases):
    return np.mean(np.std(Phases, axis=0))



def main(sim: Simulator, coeff_values: list, N=50):


    pos = sim.center_RX_position

    for item in coeff_values:
        l_h               = item[0]
        l_g               = item[1]
        TX_RX_mult_factor = item[2]


        sim.update_config('channel_modeling', 'l_h', l_h)
        sim.update_config('channel_modeling', 'l_g', l_g)
        sim.update_config('channel_modeling', 'TX_RX_mult_factor', TX_RX_mult_factor)

        sim.initialize()

        configurations = np.empty(shape=(N, sim.total_RIS_controllable_elements))
        snrs           = np.empty(shape=N)
        Hs             = np.empty(shape=(N, sim.total_RIS_elements), dtype=complex)
        Gs             = np.empty(shape=(N, sim.total_RIS_elements), dtype=complex)
        h0s            = np.empty(shape=N, dtype=complex)

        for i in tqdm(range(N)):
            # configurations[i, :] = 0
            # snrs[i]              = 0

            H, G, h0           = sim.simulate_transmission(pos)
            configuration, snr = sim.find_best_configuration(H, G, h0)
            configurations[i,:] = configuration
            snrs[i]             = snr



            Hs[i,:]             = H.flatten()
            Gs[i,:]             = G.flatten()
            h0s[i]              = h0.flatten()

            # tqdm.write("Best configuration: {} | SNR: {}".format(configuration, snr))

        title_extra = f"($ \kappa_{{h}}={l_h},\ \kappa_{{g}}={l_g}$, Direct link: {TX_RX_mult_factor!=0})"

        plot_correlation_of_elements(configurations, title_extra)
        #plt.savefig('figs/channel_infos/phase_correlations_' + title_extra + '.png')
        plt.show()

        plot_phase_elements_distributions(configurations, phase_names=['$0$', '$\pi$'], title_extra=title_extra)
        #plt.savefig('figs/channel_infos/phase_distributions_'+title_extra+'.png')
        plt.show()

        plot_snr_distribution(snrs, title_extra=title_extra)
        #plt.savefig('figs/channel_infos/snr_distributions_' + title_extra + '.png')
        #plt.show()

        plot_channels(Hs, Gs, h0s, sim.num_RIS)
        #plt.savefig('figs/channel_infos/H_values_' + title_extra + '.png')
        #plt.show()

        print(mean_stds_phases(configurations))




if __name__ == '__main__':
    sim = Simulator('setups/experiment1.ini')
    coeff_values = [
    #   (l_h, l_g,  TX_RX_coefficient)
        (5,   13.2, 1),
        (20, 50,  1),
        # (5,   13.2, 0),
        # (100, 100,  0),
    ]
    main(sim, coeff_values, N=100)