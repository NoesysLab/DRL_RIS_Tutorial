import numpy as np
from numpy import pi, cos, sin
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from typing import List, Tuple

import random


from core.surfaces import RIS
import core.channels as channels
from core.channels import RIS_RX_channel_model, TX_RIS_channel_model, TX_RX_channel_model


from utils.binary_space import BinaryEnumerator
from utils.custom_configparser import CustomConfigParser
from utils.custom_types import Vector3D, Matrix3DCoordinates
from utils.misc import ray_to_elevation_azimuth, diag_per_row, dBm_to_Watt

import configparser


from core.geometry import get_receiver_positions
from core.surfaces import RIS
from utils.custom_configparser import CustomConfigParser
from utils.custom_types import *
from utils.plotting import plot_simulation


def check_setup_satisfies_constraints(config: CustomConfigParser):

    assert config.get('RIS'  ,            'facing_direction') == 'perpendicular'
    assert config.get('setup',            'environment_type') == 'outdoor'
    assert config.get('channel_modeling', 'TX_RIS_link_type') == 'Ricean'
    assert config.get('channel_modeling', 'RIS_RX_link_type') == 'Pure LOS'
    assert config.get('channel_modeling', 'TX_RX_link_type')  == 'Rayleigh'

def load_config_from_file(filename: str)->CustomConfigParser:
    config = CustomConfigParser(interpolation=configparser.ExtendedInterpolation(),
                                allow_no_value=True,
                                inline_comment_prefixes=('#',))

    with open(filename, 'r') as f:
        config.read_file(f)

    return config


def create_setup_from_config(config: CustomConfigParser):


    check_setup_satisfies_constraints(config)


    TX_coordinates                = config.getlist ('setup', 'TX_coordinates')
    center_RX_position            = config.getlist ('setup', 'RX_grid_center')
    num_RIS                       = config.getint  ('setup', 'number_of_RIS')
    RIS_elements                  = config.getlist ('setup', 'RIS_elements',       dtype=int )
    RIS_element_groups            = config.getlist ('setup', 'RIS_element_groups', dtype=int )
    noise_power                   = config.getfloat('setup', 'noise_power')
    transmit_power                = config.getfloat('setup', 'transmit_power')
    element_dimensions            = config.getlist ('RIS'  , 'element_dimensions')
    element_gap                   = config.getlist ('RIS'  , 'element_gap')
    RIS_phase_values              = config.getlist ('RIS'  , 'phases')
    lambda_p                      = config.getfloat('channel_modeling', 'lambda_p')
    num_scatterers_clusters       = config.get('channel_modeling', 'num_scatterers_clusters')

    assert RIS_elements[0] % RIS_element_groups[0] == 0 and RIS_elements[1] % RIS_element_groups[1] == 0

    if num_scatterers_clusters == 'random':
        num_clusters = None
    else:
        num_clusters = config.getint('channel_modeling', 'num_scatterers_clusters')

    total_RIS_elements              = num_RIS * RIS_elements[0] * RIS_elements[1]
    total_RIS_controllable_elements = total_RIS_elements // (RIS_element_groups[0]* RIS_element_groups[1])


    RIS_list        = []
    RIS_coordinates = []
    for i in range(num_RIS):
        curr_RIS_coordinates = config.getlist('setup'  , 'RIS{}_coordinates'.format(i+1))

        ris = RIS(position=curr_RIS_coordinates,
                  facing_direction=TX_coordinates - center_RX_position,
                  element_grid_shape=RIS_elements,
                  element_group_size=RIS_element_groups,
                  element_dimensions=element_dimensions,
                  in_group_spacing=element_gap,
                  between_group_spacing=element_gap,
                  phase_space=('discrete', {'values': np.exp(1j*RIS_phase_values)}),
                  id_=i)

        ris.set_random_state()
        RIS_list.append(ris)
        RIS_coordinates.append(curr_RIS_coordinates)

    RIS_coordinates = np.array(RIS_coordinates)

    RX_locations    = get_receiver_positions(config.get('setup', 'RX_placement_type'),
                                            config.getint('setup', 'number_of_RX_grid_positions'),
                                            center_RX_position[0:2],
                                            config.getfloat('setup', 'RX_grid_width'),
                                            center_RX_position[2])


    return [ RIS_list,
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
             center_RX_position ]




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
    best_batch_results            = np.empty(shape=(num_batches_required,total_tunable_elements), dtype=int)
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
        best_batch_results[i,:]        = best_configuration
        best_batch_snrs[i]             = best_batch_snr


    best_snr_index_from_batches        = int(np.argmax(best_batch_snrs))
    total_best_configuration           = best_batch_results[best_snr_index_from_batches]
    total_best_snr                     = best_batch_snrs[best_snr_index_from_batches]


    return total_best_configuration, total_best_snr
















class Simulator:
    def __init__(self, setup_file: str):
        self.setup_file                          = setup_file                             # type: str
        self.config                              = load_config_from_file(self.setup_file) # type: CustomConfigParser
        self.setup_name                          = setup_file.split("/")[-1].split(".")[0]

        setup_variables                          = create_setup_from_config(self.config)

        self.RIS_list                            = setup_variables[0]                     # type: List[RIS]
        self.RX_locations                        = setup_variables[1]                     # type: Matrix3DCoordinates
        self.TX_coordinates                      = setup_variables[2]                     # type: Vector3D
        self.RIS_coordinates                     = setup_variables[3]                     # type: Matrix3DCoordinates
        self.lambda_p                            = setup_variables[4]                     # type: float
        self.num_clusters                        = setup_variables[5]                     # type: int
        self.num_RIS                             = setup_variables[6]                     # type: int
        self.total_RIS_elements                  = setup_variables[7]                     # type: int
        self.total_RIS_controllable_elements     = setup_variables[8]                     # type: int
        self.transmit_power                      = setup_variables[9]                     # type: float
        self.noise_power                         = setup_variables[10]                    # type: float
        self.center_RX_position                  = setup_variables[11]                    # type: Vector3D

        self.num_RIS_states                      = len(self.RIS_list[0].phase_space.values)   # type: int
        self.RIS_phase_values                    = self.RIS_list[0].phase_space.values        # type: np.ndarray
        self.noise_power                         = dBm_to_Watt(self.noise_power)


        self.batch_size      = self.config.getint('program_options', 'batch_size')             # type: int
        self.verbosity       = self.config.getint('program_options', 'verbosity_level')        # type: int
        self.seed            = self.config.getint('program_options', 'random_seed')            # type: int
        self.stop_iterations = self.config.getint('program_options', 'stop_after_evaluations') # type: int

        if self.seed is not None:
            np.random.seed(self.seed )
            random.seed(self.seed )

        channels.initialize_from_config(self.config)


        scatterers_variables = channels.generate_clusters(self.TX_coordinates, self.RIS_coordinates, self.lambda_p, self.num_clusters)

        self.Sc                      = scatterers_variables[0]                                # type: List[int]
        self.cluster_positions       = scatterers_variables[1]                                # type: Matrix3DCoordinates
        self.TX_clusters_distances   = scatterers_variables[2]                                # type: Matrix2D
        self.clusters_RIS_distances  = scatterers_variables[3]                                # type: Matrix3D
        self.thetas_AoA              = scatterers_variables[4]                                # type: Matrix2D
        self.phis_AoA                = scatterers_variables[5]                                # type: Matrix2D

        print("Using setup '{}'".format(self.setup_name))

        if self.verbosity >= 1:
            print("Running simulation with {} RIS and {} total elements ({} controllable)."
                  .format(self.num_RIS,
                          self.total_RIS_elements,
                          self.total_RIS_controllable_elements))

        if self.verbosity >= 2:
            self.config.print()

        if self.verbosity >= 1:
            print("Generated {} clusters with {} scatterers.".format(len(self.Sc), self.Sc))

        if self.config.getboolean('program_options', 'plot_setup'):
            plot_simulation(self.RIS_list, self.cluster_positions, self.TX_coordinates, self.center_RX_position)



    def configuration2phases(self, configurations: np.ndarray):
        assert configurations.ndim == 2
        ris = self.RIS_list[0]
        dependent_elements_per_RIS = ris.num_dependent_elements

        phase_space = ris.phase_space
        phases = phase_space.calculate_phase_shifts(configurations)
        phases = np.repeat(phases, repeats=dependent_elements_per_RIS, axis=1)
        Phi = diag_per_row(phases)
        return Phi



    def simulate_transmission(self, RX_position):
        H = channels.calculate_H(self.RIS_list, self.TX_coordinates, self.Sc, self.TX_clusters_distances,
                                 self.clusters_RIS_distances, self.thetas_AoA, self.phis_AoA)
        G, h0 = channels.calculate_G_and_h0(self.RIS_list, self.TX_coordinates, RX_position)
        
        return H, G, h0
    
    
    
    def calculate_SNR(self, H: np.ndarray, G: np.ndarray, Phi: np.ndarray, h0=None):
        if h0 is None: h0 = 0.

        if H.ndim > 1:
            H  = H[:, :, np.newaxis]
            G  = G[:, np.newaxis, :]
            h0 = np.array(h0)
            h0 = h0[:, np.newaxis, np.newaxis]

        snr = channels.compute_SNR(H, G, Phi, h0, self.noise_power)
        return snr


    def find_best_configuration(self, H: np.ndarray, G: np.ndarray, h0=None):
        if h0 is None: h0 = 0.
        return exhaustive_search(self.RIS_list, H, G, h0, self.noise_power, self.batch_size)































