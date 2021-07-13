import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from core.simulation import Simulator





def generate_data(sim, dataset, output_file, num_realizations):
    #H_prev, G_prev, h0_prev = None, None, None

    for i in tqdm(range(num_realizations)):

        H, G, h0           = sim.simulate_transmission(sim.center_RX_position)
        configuration, snr = sim.find_best_configuration(H, G, h0)

        if sim.verbosity >= 3:
            tqdm.write("Best configuration: {} | SNR: {}".format(configuration, snr))

        #H_prev, G_prev, h0_prev = np.copy(H), np.copy(G), np.copy(h0)

        dataset.add_datapoint(H, G, h0, sim.RX_train_locations[i, :], configuration, snr)

    dataset.save(output_file)
    print('Saved to: {}'.format(output_file))



def plot_data(sim, dataset, output_file):
    dataset = dataset.load(output_file)

    df_phases = pd.DataFrame(data=dataset.get('best_configuration'),
                             columns=['phase_value_{}'.format(i + 1) for i in
                                      range(sim.total_RIS_controllable_elements)])

    for i in range(sim.total_RIS_controllable_elements):
        p = dataset.get('best_configuration')[:, i].mean()
        print("Mean phase for element {:2d}: {:.2f} ± {:.2f}".format(i, p, p * (1 - p)))

    num_evaluations = dataset.shape[0]
    fig, ax = plt.subplots(figsize=(10, 7))
    xs = np.array(range(sim.total_RIS_controllable_elements))
    width = 0.35
    ax.bar(xs - width / 2, np.sum(dataset.get('best_configuration'), axis=0), width, label=r'$\pi$')
    ax.bar(xs + width / 2, num_evaluations - np.sum(dataset.get('best_configuration'), axis=0), width,
                    label='0')
    ax.set_ylabel('State frequency')
    ax.set_title('Individual element states over multiple evaluations')
    ax.set_xticks(xs)
    ax.set_xticklabels(["Element {}".format(i + 1) for i in xs], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.show()

    try:
        plt.figure()
        sns.histplot(dataset.get('best_SNR'), kde=True, stat='probability', bins=40)
        plt.show()
    except np.linalg.LinAlgError:
        pass

    plt.figure()
    plt.imshow(np.transpose(df_phases.values))
    plt.show()




if __name__ == '__main__':
    if len(sys.argv) < 2: raise RuntimeError("Expected setup configuration filename as first argument.")

    configuration_filename  = sys.argv[1]
    sim_                    = Simulator(configuration_filename)
    dataset_                = sim_.get_dataset()
    data_file               = sim_.dataSaver.get_save_filename('single_RX_position_many_realizations')

    num_realizations_       = 30


    if '--generate' in sys.argv or not os.path.isfile(data_file+'.npy'):
        print(f'Generating data with {num_realizations_} channel realizations.')
        generate_data(sim_, dataset_, data_file, num_realizations_)
    else:
        print(f'using existing dataset from {data_file}')

    plot_data(sim_, dataset_, data_file)




