from datetime import datetime

import numpy as np
from tqdm import tqdm

from core.setup import load_config_from_file, create_setup_from_config
from core.channels import generate_clusters, calculate_RX_scatterers_distances, initialize_from_config
from core.simulation import exhaustive_search, calculate_H, calculate_G_and_h0
from utils.data_handlers import SimulationDataset
from utils.plotting import plot_simulation

start_t = datetime.now()


# todo: 2. Reformat simulation vs channel code
# todo: 3. Hash config to save on directory


config = load_config_from_file('setups/setup1.ini')

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

#plot_simulation(RIS_list, cluster_positions, TX_coordinates, center_RX_position)


dataset = SimulationDataset(num_RIS, total_RIS_elements, total_RIS_controllable_elements)


for i in tqdm(range(RX_locations.shape[0])):
    H                     = calculate_H(RIS_list, TX_coordinates, Sc, TX_clusters_distances,
                                        clusters_RIS_distances, thetas_AoA, phis_AoA)
    RX_clusters_distances = calculate_RX_scatterers_distances(Sc, center_RX_position, cluster_positions)
    G, h0                 = calculate_G_and_h0(RIS_list, TX_coordinates, RX_locations[i, :])
    configuration, snr    = exhaustive_search(RIS_list, H, G, h0, noise_power, batch_size=10, show_progress_bar=False)

    print("SNR: {}".format(snr))
    print("Optimal Configuration: {}".format(configuration))

    dataset.add_datapoint(H, G, h0, RX_locations[i, :], configuration, snr)

    if i>10:
        break

dataset.save("./data/test_simulation2.npy")



end_t = datetime.now()
duration = end_t-start_t
print("Run simulation with {} RIS, {} total elements ({} controllable).  Time elapsed: {}".format(
    num_RIS, total_RIS_elements, total_RIS_controllable_elements, duration))



print(np.absolute(dataset.get('H')))

print(np.angle(dataset.get('H')))