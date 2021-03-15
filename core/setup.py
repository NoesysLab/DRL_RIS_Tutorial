import configparser
from typing import *
import numpy as np
import yaml

from configparser import ConfigParser

from core.geometry import get_receiver_positions
from core.surfaces import RIS
from utils.custom_types import *

from utils.expressions_parsing import NumericStringParser

__numericStringParser = NumericStringParser()


def parse_list(l_str: str, is_numerical=True, dtype=None):
    l_str = l_str.replace("[","").replace("]","")
    l     = l_str.split(',')
    l     = map(lambda item: item.strip(), l)
    if is_numerical:
        return np.array([parse_expr(item, dtype) for item in l])
    else:
        return l

def parse_expr(expr_str, dtype=None):
    val = __numericStringParser.eval(expr_str, parseAll=True)
    if dtype is not None:
        return dtype(val)
    else:
        return val








def check_setup_satisfies_constraints(config: ConfigParser):

    assert config.get('RIS'  ,            'facing_direction') == 'perpendicular'
    assert config.get('setup',            'environment_type') == 'outdoor'
    assert config.get('channel_modeling', 'TX_RIS_link_type') == 'Ricean'
    assert config.get('channel_modeling', 'RIS_RX_link_type') == 'Pure LOS'
    assert config.get('channel_modeling', 'TX_RX_link_type')  == 'Rayleigh'





def load_config_from_file(filename: str)->ConfigParser:
    config = ConfigParser(interpolation=configparser.ExtendedInterpolation(),
                          allow_no_value=True,
                          inline_comment_prefixes=('#',))

    with open(filename, 'r') as f:
        config.read_file(f)

    return config



'''

To print a config file use:

    cp = configparser.ConfigParser()
    cp.read_string(ini)
    with io.StringIO() as ss:
        cp.write(ss)
        ss.seek(0) # rewind
        logging.warning(ss.read())





'''




def create_setup_from_config(config: ConfigParser):


    check_setup_satisfies_constraints(config)


    TX_coordinates                = parse_list( config.get('setup', 'TX_coordinates') )
    center_RX_position            = parse_list( config.get('setup', 'RX_grid_center') )
    num_RIS                       = parse_expr( config.get('setup', 'number_of_RIS'),      dtype=int )
    RIS_elements                  = parse_list( config.get('setup', 'RIS_elements'),       dtype=int )
    RIS_element_groups            = parse_list( config.get('setup', 'RIS_element_groups'), dtype=int )
    noise_power                   = parse_expr( config.get('setup', 'noise_power'),        dtype=float)
    transmit_power                = parse_expr( config.get('setup', 'transmit_power'),     dtype=float)
    element_dimensions            = parse_list( config.get('RIS'  , 'element_dimensions') )
    element_gap                   = parse_list( config.get('RIS'  , 'element_gap'))
    RIS_phase_values              = parse_list( config.get('RIS'  , 'phases'))
    lambda_p                      = parse_expr( config.get('channel_modeling', 'lambda_p'), dtype=float)
    num_scatterers_clusters       = config.get('channel_modeling', 'num_scatterers_clusters')

    assert RIS_elements[0] % RIS_element_groups[0] == 0 and RIS_elements[1] % RIS_element_groups[1] == 0

    if num_scatterers_clusters == 'random':
        num_clusters = None
    else:
        num_clusters = parse_expr(num_scatterers_clusters, dtype=int)

    total_RIS_elements              = num_RIS * RIS_elements[0] * RIS_elements[1]
    total_RIS_controllable_elements = total_RIS_elements // (RIS_element_groups[0]* RIS_element_groups[1])


    RIS_list        = []
    RIS_coordinates = []
    for i in range(num_RIS):
        curr_RIS_coordinates = parse_list( config.get('setup'  , 'RIS{}_coordinates'.format(i+1)))

        ris = RIS(position=curr_RIS_coordinates,
                  facing_direction=TX_coordinates - center_RX_position,
                  element_grid_shape=RIS_elements,
                  element_group_size=RIS_element_groups,
                  element_dimensions=element_dimensions,
                  in_group_spacing=element_gap,
                  between_group_spacing=element_gap,
                  phase_space=('discrete', {'values': RIS_phase_values}),
                  id_=i)

        ris.set_random_state()
        RIS_list.append(ris)
        RIS_coordinates.append(curr_RIS_coordinates)

    RIS_coordinates = np.array(RIS_coordinates)

    RX_locations    = get_receiver_positions(config.get('setup', 'RX_placement_type'),
                                            parse_expr( config.get('setup', 'number_of_RX_grid_positions'), dtype=int),
                                            center_RX_position[0:2],
                                            parse_expr( config.get('setup', 'RX_grid_width'), dtype=float),
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








