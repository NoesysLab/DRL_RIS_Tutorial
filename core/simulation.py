import numpy as np
from numpy import pi, cos, sin
from tqdm import tqdm
import itertools
from typing import List, Tuple

import random

from core.surfaces import RIS
import core.channels as channels
from core.channels import RIS_RX_channel_model, TX_RIS_channel_model, TX_RX_channel_model


#from utils.binary_space import BinaryEnumerator
from utils.custom_configparser import CustomConfigParser
from utils.custom_types import Vector3D, Matrix3DCoordinates
from utils.data_handlers import SimulationDataset, DataSaver
from utils.misc import ray_to_elevation_azimuth, diag_per_row, dBm_to_Watt, dBW_to_Watt

import configparser



import numba



from core.geometry import get_receiver_positions
from core.surfaces import RIS
from utils.custom_configparser import CustomConfigParser
from utils.custom_types import *
from utils.plotting import plot_simulation


def check_setup_satisfies_constraints(config: CustomConfigParser):

    assert config.get('program_options',  'version')          == '3.0.5'         , 'Code does not support the current config version.'
    assert config.get('RIS'  ,            'facing_direction') == 'perpendicular'
    assert config.get('setup',            'environment_type') == 'outdoor'

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
    num_RIS                       = config.getint  ('setup', 'number_of_RIS')
    RIS_elements                  = config.getlist ('setup', 'RIS_elements',       dtype=int )
    RIS_element_groups            = config.getlist ('setup', 'RIS_element_groups', dtype=int )
    transmit_snr                  = config.getfloat('setup', 'transmit_SNR')
    element_dimensions            = config.getlist ('RIS'  , 'element_dimensions')
    element_gap                   = config.getlist ('RIS'  , 'element_gap')
    RIS_phase_values              = config.getlist ('RIS'  , 'phases')
    center_RX_position            = config.getlist('exhaustive_search', 'RX_grid_center')
    runs                          = config.getint('exhaustive_search', 'average_over_runs')

    assert RIS_elements[0] % RIS_element_groups[0] == 0 and RIS_elements[1] % RIS_element_groups[1] == 0

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

    RX_train_locations = get_receiver_positions(  config.get   ('exhaustive_search', 'train_RX_placement_type'),
                                                  config.getint('exhaustive_search', 'train_number_of_RX_grid_positions'),
                                                  center_RX_position[0:2],
                                                  config.getfloat('exhaustive_search', 'RX_grid_width'),
                                                  center_RX_position[2])

    RX_test_locations = get_receiver_positions(  config.get   ('exhaustive_search', 'test_RX_placement_type'),
                                                  config.getint('exhaustive_search', 'test_number_of_RX_grid_positions'),
                                                  center_RX_position[0:2],
                                                  config.getfloat('exhaustive_search', 'RX_grid_width'),
                                                  center_RX_position[2])

    return [ RIS_list,
             RX_train_locations,
             RX_test_locations,
             TX_coordinates,
             RIS_coordinates,
             num_RIS,
             total_RIS_elements,
             total_RIS_controllable_elements,
             transmit_snr,
             center_RX_position,
             runs]







@numba.njit
def nb_block(X):
    xtmp1 = np.concatenate(X[0], axis=1)
    xtmp2 = np.concatenate(X[1], axis=1)
    return np.concatenate((xtmp1, xtmp2), axis=0)


@numba.njit
def nb_block2(X):
    xtmp1 = np.hstack(X[0])
    xtmp2 = np.hstack(X[1])
    return np.vstack((xtmp1, xtmp2))


@numba.jitclass([
    ('B', numba.int8[:,:]),
    ('k', numba.int64),
    ('num_digits', numba.int64),
    ('max_number', numba.float64),
    ('curr_padding', numba.int64),
    ('_curr_num', numba.int64),
])
class BinaryEnumerator:


    def __init__(self, num_digits):
        self.B            = np.array([[0],[1]], dtype=np.byte)
        self.k            = 1
        self.num_digits   = num_digits
        self.max_number   = np.power(2, num_digits)-1
        self.curr_padding = self.num_digits - self.k

        self._curr_num   = 0


    def _expand_array(self):

        self.B = nb_block2(((np.zeros((2 ** self.k, 1), dtype=np.byte), self.B),
                           (np.ones((2 ** self.k, 1), dtype=np.byte), self.B)))

        self.k += 1
        self.curr_padding = self.num_digits - self.k


    # def __iter__(self):
    #     self._curr_num = 0
    #     return self


    def next(self):
        if self._curr_num > self.max_number:
            raise StopIteration


        if self._curr_num + 1 > np.power(2, self.k):
            self._expand_array()


        num = self.B[self._curr_num, :]
        self._curr_num += 1


        if self.curr_padding>0:
            num = np.concatenate( ( np.zeros(self.curr_padding, dtype=np.byte), num ) )

        return num





@numba.vectorize(nopython=True)
def to_complex(x):
    return x+0j






@numba.njit(fastmath=True)
def exhaustive_search(H, G, h0, transmit_snr,
                      total_tunable_elements,
                      total_dependent_elements,
                      phase_values, )->Tuple[np.ndarray, float]:

    num_discrete_states     = len(phase_values)
    configurations_iterator = BinaryEnumerator(total_tunable_elements)
    num_configurations      = int(2**total_tunable_elements)

    best_snr                = 0.
    best_configuration      = np.zeros(num_discrete_states, dtype=np.byte)


    assert num_discrete_states == 2, 'Currently supporting only two discrete states due to custom enumeration of the configuration/phase space.'


    for i in range(num_configurations):

        configuration                  = configurations_iterator.next().flatten()
        phase                          = phase_values[configuration]
        Phi                            = np.repeat(phase, repeats=total_dependent_elements)
        Phi2                           = to_complex(Phi).flatten()
        prod                           = (G.T*H).flatten()
        channel_reflected              = np.dot(prod, Phi2) + h0
        channel_reflected2             = channel_reflected[0,0]
        channel_mag                    = np.power( np.absolute(channel_reflected2), 2)
        snr                            = channel_mag * transmit_snr

        if snr > best_snr:
            best_snr           = snr
            best_configuration = configuration

    return best_configuration, best_snr




class Simulator:
    def __init__(self, setup_file: str):
        self.setup_file                          = setup_file                                                           # type: str
        self.config                              = load_config_from_file(self.setup_file)                               # type: CustomConfigParser
        self.setup_name                          = setup_file.split("/")[-1].split(".")[0]                              # type: str

        setup_variables                          = create_setup_from_config(self.config)

        self.RIS_list                            = setup_variables[0]                                                   # type: List[RIS]
        self.RX_train_locations                  = setup_variables[1]                                                   # type: Matrix3DCoordinates
        self.RX_test_locations                   = setup_variables[2]                                                   # type: Matrix3DCoordinates
        self.TX_coordinates                      = setup_variables[3]                                                   # type: Vector3D
        self.RIS_coordinates                     = setup_variables[4]                                                   # type: Matrix3DCoordinates
        self.num_RIS                             = setup_variables[5]                                                   # type: int
        self.total_RIS_elements                  = setup_variables[6]                                                   # type: int
        self.total_RIS_controllable_elements     = setup_variables[7]                                                   # type: int
        self.transmit_SNR_db                     = setup_variables[8]                                                   # type: float
        self.center_RX_position                  = setup_variables[9]                                                   # type: Vector3D
        self.runs                                = setup_variables[10]                                                  # type: int

        self.num_RIS_states                      = len(self.RIS_list[0].phase_space.values)                             # type: int
        self.RIS_phase_values                    = self.RIS_list[0].phase_space.values                                  # type: np.ndarray
        self.transmit_SNR                        = dBW_to_Watt(self.transmit_SNR_db)                                    # type: float
        self.total_dependent_elements            = self.total_RIS_elements // self.total_RIS_controllable_elements      # type: int

        self.batch_size                          = self.config.getint('program_options', 'batch_size')                  # type: int
        self.verbosity                           = self.config.getint('program_options', 'verbosity_level')             # type: int
        self.seed                                = self.config.getint('program_options', 'random_seed')                 # type: int
        self.stop_iterations                     = self.config.getint('program_options', 'stop_after_evaluations')      # type: int

        self.dataSaver                           = DataSaver(self.setup_name,
                                                             save_dir=self.config.get('program_options',
                                                                                      'output_directory_root')
                                                             ).set_configuration(self.config)

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        channels.initialize_from_config(self.config)

        print("Using setup '{}'".format(self.setup_name))

        if self.verbosity >= 1:
            print("Running simulation with {} RIS and {} total elements ({} controllable)."
                  .format(self.num_RIS,
                          self.total_RIS_elements,
                          self.total_RIS_controllable_elements))

        if self.verbosity >= 2:
            self.config.print()


        if self.config.getboolean('program_options', 'plot_setup'):
            plot_simulation(self.RIS_list, None, self.TX_coordinates, self.center_RX_position)


    def get_dataset(self) -> SimulationDataset:
        return SimulationDataset(self.num_RIS, self.total_RIS_elements,
                                 self.total_RIS_controllable_elements)


    def get_save_dir(self, filename=None)->str:
        filename = '' if not filename else filename
        return self.dataSaver.get_save_filename(filename)

    def get_exhaustive_results_filename(self, mode):
        assert mode in {'test', 'train', 'test_avg', 'train_avg'}
        if '_avg' in mode:
            is_avg = True
            mode = mode.replace("_avg","")
        else:
            is_avg = False
        output_filename = self.config.get('exhaustive_search', mode + '_output_file_name')
        if is_avg:
            output_filename += '_avg'
        output_filename_with_path = self.dataSaver.get_save_filename(output_filename)
        return output_filename_with_path

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
        H = channels.calculate_H(self.RIS_list, self.TX_coordinates)
        G, h0 = channels.calculate_G_and_h0(self.RIS_list, self.TX_coordinates, RX_position)
        
        return H, G, h0
    
    
    
    def calculate_SNR(self, H: np.ndarray, G: np.ndarray, Phi: np.ndarray, h0=None):
        if h0 is None: h0 = 0.

        if H.ndim > 1:
            H  = H[:, :, np.newaxis]
            G  = G[:, np.newaxis, :]
            h0 = np.array(h0)
            h0 = h0[:, np.newaxis, np.newaxis]

        snr = channels.compute_SNR(H, G, Phi, h0, self.transmit_SNR)
        return snr


    def find_best_configuration(self, H: np.ndarray, G: np.ndarray, h0=None):
        if h0 is None: h0 = 0.
        return exhaustive_search(H, G, h0,
                                 self.transmit_SNR,
                                 self.total_RIS_controllable_elements,
                                 self.total_dependent_elements,
                                 self.RIS_list[0].phase_space.values
                                 )






























