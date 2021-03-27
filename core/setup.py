import configparser
from typing import *
import numpy as np
import yaml



from core.geometry import get_receiver_positions
from core.surfaces import RIS
from utils.custom_configparser import CustomConfigParser
from utils.custom_types import *











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








