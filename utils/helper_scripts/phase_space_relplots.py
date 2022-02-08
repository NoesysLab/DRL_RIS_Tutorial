import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_correlation_of_elements(Phases):
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
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")



def plot_phase_elements_distributions(Phases, phase_names=None):
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

    plt.title(f"Distribution of phase values per element ({len(df)} trials)")
    plt.xlabel("RIS element")
    plt.ylabel("frequency")
    plt.hlines(.5, -.5, len(df.columns)-.5, colors='k', linestyles='--')
    plt.legend()



def double_mae_from_05(Phases):
    avg_values = np.mean(Phases, axis=0)
    return np.sum(np.abs(avg_values-0.5)) / len(avg_values) * 2


if __name__ == '__main__':

    def main():
        F = np.random.randint(0, 2, size=(100,8), )
        F = np.hstack([F,F])

        plt.figure(figsize=(10,6))
        plot_correlation_of_elements(F)
        plt.show()

        plot_phase_elements_distributions(F)
        plt.show()

        mae = double_mae_from_05(F)
        print(mae)

    main()