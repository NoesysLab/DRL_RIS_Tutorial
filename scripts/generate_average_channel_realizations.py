from datetime import datetime
import sys
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


from core.setup import load_config_from_file, create_setup_from_config
from core.simulation import exhaustive_search, calculate_H, calculate_G_and_h0, generate_clusters, calculate_RX_scatterers_distances, initialize_from_config
from utils.data_handlers import SimulationDataset, DataSaver
from utils.misc import dBm_to_Watt
from utils.plotting import plot_simulation

start_t = datetime.now()




configuration_filename = 'setups/setup1.ini'
SETUP = configuration_filename.split("/")[-1].split(".")[0]


print("Using setup '{}'".format(SETUP))



config    = load_config_from_file(configuration_filename)


batch_size      = config.getint('program_options', 'batch_size')
verbosity       = config.getint('program_options', 'verbosity_level')
seed            = config.getint('program_options', 'random_seed')
stop_iterations = config.getint('program_options', 'stop_after_evaluations')


if seed:
    np.random.seed(seed)
    random.seed(seed)


initialize_from_config(config)



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






noise_power = dBm_to_Watt(noise_power)

[Sc,
 cluster_positions,
 TX_clusters_distances,
 clusters_RIS_distances,
 thetas_AoA,
 phis_AoA                         ] = generate_clusters(TX_coordinates,
                                                        RIS_coordinates,
                                                        lambda_p,
                                                        num_clusters)


if verbosity >= 1:
    print("Running simulation with {} RIS and {} total elements ({} controllable).".format(num_RIS, total_RIS_elements, total_RIS_controllable_elements))


if verbosity >= 2:
    config.print()


if verbosity >= 1:
    print("Generated {} clusters with {} scatterers.".format(len(Sc), Sc))


inp_dataset = SimulationDataset(num_RIS, total_RIS_elements, total_RIS_controllable_elements)\
    .load('data/simulations/setup1/987b6478417007f985e675cd/exhaustive_search.npy')


out_dataset = SimulationDataset(num_RIS, total_RIS_elements, total_RIS_controllable_elements)

num_samples = 400

RX_positions        = inp_dataset.get('RX_position')
best_configurations = inp_dataset.get('best_configuration')
best_SNRs           = inp_dataset.get('best_SNR')


H_samples  = np.empty((num_samples, total_RIS_elements), dtype=complex)
G_samples  = np.empty((num_samples, total_RIS_elements), dtype=complex)
h0_samples = np.empty(num_samples, dtype=complex)


for i in tqdm(range(len(RX_positions))):


    for j in range(num_samples):
        H = calculate_H(RIS_list, TX_coordinates, Sc, TX_clusters_distances,
                        clusters_RIS_distances, thetas_AoA, phis_AoA)
        G, h0 = calculate_G_and_h0(RIS_list, TX_coordinates, RX_locations[i, :])

        H_samples[j, :] = H.flatten()
        G_samples[j, :] = G.flatten()
        h0_samples[j]  = h0.flatten()

    H_mean = H_samples.mean(axis=0).flatten()
    G_mean = G_samples.mean(axis=0).flatten()
    h0_mean = G_samples.mean().flatten()

    out_dataset.add_datapoint(H_mean, G_mean, h0_mean, RX_positions[i,:],best_configurations[i,:], best_SNRs[i])


out_dataset.save('data/simulations/setup1/987b6478417007f985e675cd/simulation_mean_channels')
