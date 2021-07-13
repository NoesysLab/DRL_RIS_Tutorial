import os
from typing import *

from utils.binary_space import BinaryEnumerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf


import numpy as np
from tqdm import tqdm


from deprecated.channels import Channel
from core.surfaces import RIS



is_tf_2 = int(tf.__version__.split(".")[0]) == 2

if not is_tf_2:
    tf.enable_eager_execution()


def exhaustive_snr_search(ch: Channel, ris_list: List[RIS], batch_size=2 ** 11, show_progress_bar=False):


    total_tunable_elements               = sum([ris.num_tunable_elements for ris in ris_list])
    dependent_elements_per_RIS           = ris_list[0].num_dependent_elements
    discrete_states                      = range(ris_list[0].state_space.num_values)
    phase_space                          = ris_list[0].phase_space
    combined_state_space_elements        = int(len(discrete_states)) ** int(total_tunable_elements)
    num_transmissions                    = combined_state_space_elements
    K                                    = sum([ris.total_elements for ris in ris_list])
    num_batches_required                 = int(np.ceil(num_transmissions / batch_size))
    last_batch_size                      = batch_size if num_transmissions % batch_size == 0 else num_transmissions % batch_size
    best_batch_results                   = []
    best_batch_snrs                      = np.empty(shape=(num_batches_required,))
    batch_indices                        = range(num_batches_required)

    possible_configurations              = BinaryEnumerator(batch_size, total_tunable_elements)

    ris_element_coordinates              = ch.get_combined_elements_coordinates(ris_list)


    ch.TX_RIS_link.initialize( [ch.tx_position]       , ris_element_coordinates )
    ch.RX_RIS_link.initialize( ris_element_coordinates, [ch.rx_position]        )
    ch.TX_RX_link .initialize( [ch.tx_position]       , [ch.rx_position]        )





    if show_progress_bar: batch_indices = tqdm(batch_indices, leave=True)

    for i in tqdm(batch_indices):

        batch_transmissions       = batch_size if i != num_batches_required - 1 else last_batch_size
        batch_configurations      = next(possible_configurations)
        batch_phases              = phase_space.calculate_phase_shifts(batch_configurations)
        batch_phases              = np.repeat(batch_phases, repeats=dependent_elements_per_RIS, axis=1)



        H                         = np.empty(shape=(batch_transmissions, K, 1), dtype=np.complex)
        G                         = np.empty(shape=(batch_transmissions, 1, K), dtype=np.complex)
        h                         = np.empty(shape=(batch_transmissions, 1, 1), dtype=np.complex)
        Phi                       = np.empty(shape=(batch_transmissions, K, K), dtype=np.complex)


        for j in range(batch_transmissions):

            H[j, :, :]            = ch.TX_RIS_link.get_transmission_matrix()  # shape: (K,1)
            G[j, :, :]            = ch.RX_RIS_link.get_transmission_matrix()  # shape: (1,K)
            h[j, :, :]            = ch.TX_RX_link.get_transmission_matrix()   # shape: (1,1)
            Phi[j, :, :]          = np.diag(batch_phases[j, :])               # shape: (K,K)



        all_snr                   = ch._calculate_SNR(H, Phi, G, h).flatten()

        best_configuration_index  = np.argmax(all_snr)
        snr                       = all_snr[best_configuration_index]

        best_configuration        = batch_configurations[best_configuration_index]
        angle_TX                  = np.angle(H[best_configuration_index, :, :])
        mag_TX                    = np.abs(H[best_configuration_index, :, :])
        angle_RX                  = np.angle(G[best_configuration_index, :, :])
        mag_RX                    = np.abs(G[best_configuration_index, :, :])

        best_batch_snrs[i]        = snr
        best_batch_results.append( (best_configuration, snr, angle_TX, mag_TX, angle_RX, mag_RX, best_configuration_index) )


    best_snr_index_from_batches   = int(np.argmax(best_batch_snrs))

    return best_batch_results[best_snr_index_from_batches]                 # best_configuration, snr, angle_TX, mag_TX, angle_RX, mag_RX,

