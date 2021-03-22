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





if len(sys.argv) < 2:
    raise RuntimeError("Expected setup configuration filename as first argument.")
else:
    configuration_filename = sys.argv[1]
    SETUP = configuration_filename.split("/")[-1].split(".")[0]


print("Using setup '{}'".format(SETUP))



config    = load_config_from_file(configuration_filename)
dataSaver = DataSaver(SETUP, config.get('program_options', 'output_directory_root')).set_configuration(config)


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




if config.getboolean('program_options', 'plot_setup'):
    plot_simulation(RIS_list, cluster_positions, TX_coordinates, center_RX_position)


dataset = SimulationDataset(num_RIS, total_RIS_elements, total_RIS_controllable_elements)


iterator = range(RX_locations.shape[0])
if verbosity >= 2:
    print("\nExhaustive search:")
    iterator = tqdm(iterator)


for i in iterator:
    if stop_iterations is not None and i>= stop_iterations: break

    H                     = calculate_H(RIS_list, TX_coordinates, Sc, TX_clusters_distances,
                                        clusters_RIS_distances, thetas_AoA, phis_AoA)
    G, h0                 = calculate_G_and_h0(RIS_list, TX_coordinates, RX_locations[i, :])
    configuration, snr    = exhaustive_search(RIS_list, H, G, h0, noise_power, batch_size=batch_size, show_progress_bar=False)

    if verbosity >= 3:
        tqdm.write("Best configuration: {} | SNR: {}".format(configuration, snr))

    dataset.add_datapoint(H, G, h0, RX_locations[i, :], configuration, snr)






dataset.save(dataSaver.get_save_filename(config.get('program_options', 'output_file_name')),
             formats=config.getlist('program_options', 'output_extensions', is_numerical=False))



end_t = datetime.now()
duration = end_t-start_t
if verbosity >= 1:
    print("Finished. Time elapsed: {}".format(duration))

print("Saved to {} .".format(dataSaver.get_save_filename('')))