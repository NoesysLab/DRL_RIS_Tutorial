import numpy as np
from numpy import pi, cos, sin
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from typing import List, Tuple


from core.surfaces import RIS
import core.channels as channels
from core.channels import RIS_RX_channel_model, TX_RIS_channel_model, TX_RX_channel_model


from utils.binary_space import BinaryEnumerator
from utils.custom_configparser import CustomConfigParser
from utils.custom_types import Vector3D, Matrix3DCoordinates
from utils.misc import ray_to_elevation_azimuth, diag_per_row



rng = None

def initialize_from_config(config: CustomConfigParser):
    channels.initialize_from_config(config)
    global rng
    seed = config.getint('program_options', 'random_seed')
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()








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

        batch_snrs                     = channels.compute_SNR(H, G, Phi, h0, noise_power)
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






def _generate_scatterers_positions(C, Sc, Smax, TX_coordinates, RIS_Coordinates, phi_TX, theta_TX):
    y_bounds = [np.min(RIS_Coordinates[:, 1])+0.01, np.max(RIS_Coordinates[:, 1])-0.01]
    x_bounds = [TX_coordinates[0]+0.01, np.min(RIS_Coordinates[:, 0])-0.01]
    z_bounds = [0, TX_coordinates[2]-0.01]

    bounds = np.array([x_bounds, y_bounds, z_bounds])

    cluster_positions = np.zeros((C, Smax, 3))

    min_TX_RIS_dist = np.min(np.linalg.norm(TX_coordinates - RIS_Coordinates, axis=1))


    for c in range(C):

        cluster_centroid_coords = rng.uniform(low=bounds[:,0], high=bounds[:,1])

        for s in range(Sc[c]):

            rotation_matrix     = [cos(theta_TX[c][s])*cos(phi_TX[c][s]), cos(theta_TX[c][s])*sin(phi_TX[c][s]), sin(theta_TX[c][s])]
            scatterer_positions = np.array(rotation_matrix) * rng.rand(3) + cluster_centroid_coords
            scatterer_positions = np.clip(scatterer_positions, a_min=bounds[:,0], a_max=bounds[:,1])


            cluster_positions[c, s, :] = scatterer_positions #[x, y, z]



    return cluster_positions



def _calculate_RIS_scatterers_distances_and_angles(C, Sc, Smax, RIS_Coordinates, cluster_positions):
    num_RIS = RIS_Coordinates.shape[0]

    clusters_RIS_distances = np.zeros((C, Smax, num_RIS))
    thetas_AoA             = np.zeros((C, Smax, num_RIS))
    phis_AoA               = np.zeros((C, Smax, num_RIS))

    for r in range(num_RIS):

        x_RIS = RIS_Coordinates[r,0]
        y_RIS = RIS_Coordinates[r,1]
        z_RIS = RIS_Coordinates[r,2]

        for c in range(C):
            for s in range(Sc[c]):
                x,y,z                         = cluster_positions[c,s,:]
                b_c_s                         = np.linalg.norm(RIS_Coordinates[r, :] - cluster_positions[c,s,:])
                clusters_RIS_distances[c,s,r] = b_c_s
                thetas_AoA[c,s,r]             = np.sign(z - z_RIS) * np.arcsin( np.abs(z_RIS - z) / b_c_s )
                phis_AoA[c,s,r]               = np.sign(x_RIS - x) * np.arctan( np.abs(x_RIS - x) / np.abs(y_RIS - y) )

    return clusters_RIS_distances, thetas_AoA, phis_AoA



def calculate_RX_scatterers_distances(Sc, RX_coordinates, cluster_positions):
    RX_clusters_distances = np.linalg.norm(RX_coordinates[None, None, :] - cluster_positions, axis=2)  # Shape (C, Sc)

    C = len(Sc)
    Smax = np.max(Sc)

    for c in range(C):
        for s in range(Sc[c], Smax):
            RX_clusters_distances[c, s] = 0

    return RX_clusters_distances








def generate_clusters(TX_coordinates : Vector3D,
                      RIS_Coordinates: Matrix3DCoordinates,
                      lambda_p       : float,
                      num_clusters=None):

    # assuming TX is on the yz plane and all RIS on the xz plane

    if num_clusters is None:
        C = np.maximum(2, rng.poisson(lambda_p))
    else:
        C = num_clusters

    Sc            = rng.randint(1, 30, size=C)
    Smax          = np.max(Sc)


    mean_phi_TX   = rng.uniform(-pi/2, pi/2, size=C)
    mean_theta_TX = rng.uniform(-pi/4, pi/4, size=C)

    phi_TX        = [rng.laplace(mean_phi_TX[c]  , 5*pi/180, size=Sc[c]) for c in range(C)]
    theta_TX      = [rng.laplace(mean_theta_TX[c], 5*pi/180, size=Sc[c]) for c in range(C)]


    cluster_positions = _generate_scatterers_positions(C, Sc, Smax, TX_coordinates, RIS_Coordinates, phi_TX, theta_TX)

    TX_clusters_distances = np.linalg.norm(TX_coordinates[None,None,:]-cluster_positions, axis=2) # Shape (C, Sc)

    for c in range(C):
        for s in range(Sc[c], Smax):
            TX_clusters_distances[c,s] = 0


    clusters_RIS_distances,\
    thetas_AoA,\
    phis_AoA = _calculate_RIS_scatterers_distances_and_angles(C, Sc, Smax, RIS_Coordinates, cluster_positions)

    return Sc, cluster_positions, TX_clusters_distances, clusters_RIS_distances, thetas_AoA, phis_AoA







