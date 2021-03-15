import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from typing import List, Tuple


from core.surfaces import RIS
from core.channels import generate_clusters, RIS_RX_channel_model, TX_RIS_channel_model, \
    TX_RX_channel_model, calculate_RX_scatterers_distances


from utils.binary_space import BinaryEnumerator
from utils.data_handlers import SimulationDataset
from utils.misc import ray_to_elevation_azimuth, diag_per_row


def exhaustive_search(ris_list: List[RIS],
                      H,
                      G,
                      h0,
                      noise_power,
                      batch_size=2**11,
                      show_progress_bar=False)->Tuple[np.ndarray, float]:

    total_tunable_elements        = sum([ris.num_tunable_elements for ris in ris_list])
    dependent_elements_per_RIS    = ris_list[0].num_dependent_elements
    discrete_states               = range(ris_list[0].state_space.num_values)
    phase_space                   = ris_list[0].phase_space
    combined_state_space_elements = int(len(discrete_states)) ** int(total_tunable_elements)
    num_transmissions             = combined_state_space_elements
    K                             = sum([ris.total_elements for ris in ris_list])
    num_batches_required          = int(np.ceil(num_transmissions / batch_size))
    last_batch_size               = batch_size if num_transmissions % batch_size == 0 else num_transmissions % batch_size
    possible_configurations       = BinaryEnumerator(batch_size, total_tunable_elements)
    best_batch_results            = np.empty(shape=(num_batches_required,), dtype=object)
    best_batch_snrs               = np.empty(shape=(num_batches_required,))
    batch_indices                 = range(num_batches_required)

    if show_progress_bar:
        batch_indices = tqdm(batch_indices, leave=True)


    for i in batch_indices:

        batch_transmissions            = batch_size if i != num_batches_required-1 else last_batch_size
        batch_configurations           = next(possible_configurations)
        batch_phases                   = phase_space.calculate_phase_shifts(batch_configurations)
        batch_phases                   = np.repeat(batch_phases, repeats=dependent_elements_per_RIS, axis=1)
        Phi                            = diag_per_row(batch_phases)

        complete_channel_coefficients  = G @ Phi @ H + h0
        batch_snrs                     = np.power(np.absolute(complete_channel_coefficients), 2) / noise_power
        batch_snrs                     = batch_snrs.flatten()

        best_batch_configuration_index = np.argmax(batch_snrs)
        best_batch_snr                 = batch_snrs[best_batch_configuration_index]
        best_configuration             = batch_configurations[best_batch_configuration_index]
        best_batch_results[i]          = best_configuration
        best_batch_snrs[i]             = best_batch_snr


    best_snr_index_from_batches        = int(np.argmax(best_batch_snrs))
    total_best_configuration           = best_batch_results[best_snr_index_from_batches]
    total_best_snr                     = best_batch_snrs[best_snr_index_from_batches]


    return total_best_configuration, total_best_snr









def calculate_H(ris_list: List[RIS],
                TX_location,
                Sc,
                TX_scatterers_distances,
                scatterers_RIS_distances,
                thetas_AoA,
                phis_AoA,):

    K = sum([ris.total_elements for ris in ris_list])

    H = []

    for i, ris in enumerate(ris_list):
        dist_TX_RIS = np.linalg.norm(TX_location - ris.position)
        theta_RIS_TX, phi_RIS_TX = ray_to_elevation_azimuth(TX_location, ris.position)

        h = TX_RIS_channel_model(ris.total_elements,
                                 Sc,
                                 thetas_AoA[:, :, i],
                                 phis_AoA[:, :, i],
                                 TX_scatterers_distances + scatterers_RIS_distances[:, :, i],
                                 dist_TX_RIS,
                                 theta_RIS_TX,
                                 phi_RIS_TX,
                                 ris.element_spacing,
                                 LOS_component_exists=True)
        H.append(h)
    H = np.array(H).reshape((K, 1))
    return H





def calculate_G_and_h0(ris_list: List[RIS],
                       TX_location,
                      RX_location,):

    TX_location = TX_location.reshape(1, 3)


    K = sum([ris.total_elements for ris in ris_list])


    G = []


    for i, ris in enumerate(ris_list):

        dist_RIS_RX = np.linalg.norm(RX_location - ris.position)
        theta_RIS_RX, phi_RIS_RX = ray_to_elevation_azimuth(ris.position, RX_location)

        g = RIS_RX_channel_model(ris.total_elements,
                                 dist_RIS_RX,
                                 theta_RIS_RX,
                                 phi_RIS_RX,
                                 ris.element_spacing)
        G.append(g)



    TX_RX_distance = np.linalg.norm(TX_location - RX_location)
    h_SISO         = TX_RX_channel_model(TX_RX_distance, wall_exists=True)

    # h_SISO = TX_RX_channel_model(Sc, TX_scatterers_distances, scatterers_RIS_distances[:,:,i], scatterers_RX_distances, TX_RX_distance, LOS_component_exists=False, wall_exists=True)




    G  = np.array(G).reshape(1, K)
    h0 = np.array(h_SISO).reshape((1, 1))

    return G, h0











# if __name__ == '__main__':
#
#     start_t = datetime.now()
#
#     isWall = True
#     Power = 1
#
#
#
#     setup = Setup.from_config({'test': {
#         'num_RIS'                : 2,#4,
#         'RIS_coordinates'        : [[15,25,2], [15, 35 ,2], ],#[25, 25, 2], [25,35,2]],
#         'RIS_elements'           : (2,2),
#         'element_dimensions'     : (0.3,0.01),
#         'RIS_element_groups'     : (1,1),
#         'RIS_phase_values'       : [np.exp(1j*0), np.exp(1j*1)],
#         'TX_locations'           : [0,30,2],
#         'TX_RIS_link_mult_factor': Power,
#         'RX_RIS_link_mult_factor': Power,
#         'TX_RX_link_mult_factor' : Power,
#         'TX_RX_link_is_LOS'      : not isWall,
#         'train_RX_square_center' : (20,30),
#         'train_RX_square_width'  : 1,
#         'train_RX_num_positions' : 5000,
#         'train_RX_height'        : 1,
#         'noise_power'            : 100,
#         }})
#
#     RIS_list,\
#     TX_RIS_link_info, \
#     RX_RIS_link_info, \
#     TX_RX_link_info, \
#     train_RX_locations, \
#     test_RX_locations, \
#     center_RX_position = initialize_simulation_from_setup(setup)
#
#     from utils.plotting import plot_setup_3D, plot_positions, grid_plot_params
#
#     Sc, \
#     cluster_positions, \
#     TX_clusters_distances, \
#     clusters_RIS_distances, \
#     thetas_AoA, \
#     phis_AoA = generate_clusters(setup.TX_locations.reshape(-1), setup.RIS_coordinates, lambda_p=1.9, num_clusters=4)
#
#
#
#     scatterers_positions = cluster_positions.reshape((-1,3)) # Shape (C*Smax, 3)
#     scatterers_positions = scatterers_positions[np.all(scatterers_positions != 0, axis=1)]
#     #print(scatterers_positions)
#
#
#
#     params = grid_plot_params.copy()
#     #params['zlims'] = [0, 3]
#     params['color_by_height'] = False
#     plot_setup_3D(RIS_list, setup.TX_locations.reshape((1,3)), center_RX_position.reshape(1,3), params=params, scatterers_positions=scatterers_positions)
#     plot_positions(np.array([ris.position for ris in RIS_list]), setup.TX_locations.reshape((1, 3)), center_RX_position.reshape(1, 3),)
#     #plt.show()
#
#     # H = calculate_H(RIS_list, setup.TX_locations.reshape(-1), TX_clusters_distances, clusters_RIS_distances, thetas_AoA,
#     #                 phis_AoA)
#
#
#     total_RIS_elements = setup.num_RIS*setup.RIS_elements[0]*setup.RIS_elements[1]
#     total_RIS_controllable_elements = total_RIS_elements // (setup.RIS_element_groups[0] * setup.RIS_element_groups[1])
#     dataset = SimulationDataset(setup.num_RIS, total_RIS_elements, total_RIS_controllable_elements)
#
#     for i in tqdm(range(train_RX_locations.shape[0])):
#         H                     = calculate_H(RIS_list, setup.TX_locations.reshape(-1), TX_clusters_distances,
#                                             clusters_RIS_distances, thetas_AoA, phis_AoA)
#         RX_clusters_distances = calculate_RX_scatterers_distances(Sc, center_RX_position, cluster_positions)
#         G, h0                 = calculate_G_and_h0(RIS_list, setup.TX_locations.reshape(1,3), train_RX_locations[i,:])
#         configuration, snr    = exhaustive_search(RIS_list, H, G, h0, setup.noise_power, batch_size=1, show_progress_bar=False)
#
#         print("SNR: {}".format(snr))
#         print("Optimal Configuration: {}".format(configuration))
#
#         dataset.add_datapoint(H, G, h0, train_RX_locations[i,:], configuration, snr)
#
#
#     dataset.save("./data/test_simulation.npy")
#
#     end_t = datetime.now()
#     duration = end_t-start_t
#     print("Run simulation with {} RIS, {} phases, {} elements grouped in {}.  Time elapsed: {}".format(
#         setup.num_RIS, len(setup.RIS_phase_values), setup.RIS_elements, setup.RIS_element_groups, duration))
#
#
#
