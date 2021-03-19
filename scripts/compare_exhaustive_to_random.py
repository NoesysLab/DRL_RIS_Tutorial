from datetime import datetime
import sys
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


from core.channels import compute_SNR
from core.setup import load_config_from_file, create_setup_from_config
from core.simulation import exhaustive_search, calculate_H, calculate_G_and_h0, generate_clusters, calculate_RX_scatterers_distances, initialize_from_config
from core.surfaces import RIS
from utils.data_handlers import SimulationDataset, DataSaver
from utils.misc import diag_per_row
from utils.plotting import plot_simulation


if len(sys.argv) < 2:
    raise RuntimeError("Expected setup configuration filename as first argument.")
else:
    configuration_filename = sys.argv[1]
    SETUP = configuration_filename.split("/")[-1].split(".")[0]






config    = load_config_from_file(configuration_filename)
dataSaver = DataSaver(SETUP, config.get('program_options', 'output_directory_root')).set_configuration(config)

dataset_filename = dataSaver.get_save_filename(config.get('program_options', 'output_file_name'))+".npy"

print("Using setup '{}'. Loading from '{}'".format(SETUP, dataset_filename))


batch_size      = config.getint('program_options', 'batch_size')
verbosity       = config.getint('program_options', 'verbosity_level')
seed            = config.getint('program_options', 'random_seed')
stop_iterations = config.getint('program_options', 'stop_after_evaluations')




if seed:
    np.random.seed(seed)
    random.seed(seed)


[RIS_list,
 RX_locations,
 TX_coordinates,
 RIS_coordinates,
 lambda_p,
 num_clusters,
 num_RIS,
 total_RIS_elements,
 total_RIS_controllable_elements,
 transmit_power,
 noise_power,
 center_RX_position              ] = create_setup_from_config(config)

[Sc,
 cluster_positions,
 TX_clusters_distances,
 clusters_RIS_distances,
 thetas_AoA,
 phis_AoA                         ] = generate_clusters(TX_coordinates,
                                                        RIS_coordinates,
                                                        lambda_p,
                                                        num_clusters)


initialize_from_config(config)



ris = RIS_list[0] # type: RIS

dataset = SimulationDataset(num_RIS, total_RIS_elements, total_RIS_controllable_elements).load(dataset_filename)




def configuration2phases(configurations: np.ndarray):
    assert configurations.ndim == 2
    dependent_elements_per_RIS = ris.num_dependent_elements

    phase_space  = ris.phase_space
    phases       = phase_space.calculate_phase_shifts(configurations)
    phases       = np.repeat(phases, repeats=dependent_elements_per_RIS, axis=1)
    Phi          = diag_per_row(phases)
    return Phi



RX_positions = dataset.get('RX_position')


H  = dataset.get('H')
G  = dataset.get('G')
h0 = dataset.get('h')


H = H[:,:, np.newaxis]
G = G[:, np.newaxis, :]
h0 = h0[:, np.newaxis, np.newaxis]

configurations_best = dataset.get('best_configuration')
SNRs_best           = dataset.get('best_SNR')


num_RIS_states        = len(ris.phase_space.values)
configurations_random = np.random.randint(0, num_RIS_states, size=configurations_best.shape)
Phi_random            = configuration2phases(configurations_random)


# H = H[0,:,:]
# G = G[0,:,:]
# h0 = h0[0]
# Phi_random = Phi_random[0,:,:]

SNRs_random           = compute_SNR(H, G, Phi_random, h0, noise_power).flatten()



Xs = np.arange(0, dataset.shape[0])


fig, ax = plt.subplots()
ax.plot(Xs, SNRs_best, label='Exhaustive search')
ax.plot(Xs, SNRs_random, label='Random phases')
plt.legend()

ax.fill_between(Xs, SNRs_best, SNRs_random, alpha=0.5, color='gray')

ax.set_xlabel('RX position')
ax.set_ylabel('SNR (strange units)')
plt.show()


fig, ax = plt.subplots()
colors = SNRs_random - SNRs_best
im = ax.scatter(RX_positions[:,0], RX_positions[:,1], c=colors)
plt.colorbar(im)
plt.show()

