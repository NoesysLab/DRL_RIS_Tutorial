import numpy as np
from scipy import stats

from core.setup import Setup, initialize_simulation_from_setup
from core.surfaces import RIS
from core.channels import Channel, RayleighFadeLink
from core.exhaustive_phase_search import find_RIS_configuration_that_maximizes_SNR
import core.globals as globals
from utils.plotting import plot_positions


import matplotlib.pyplot as plt

if __name__ == '__main__':



    dh           = 20
    x1 = x2 = y1 = 25
    y2           = 35
    isWall       = False
    Power        = 1

    setup = Setup.from_config({'test': {
        'num_RIS'                : 4,
        'RIS_coordinates'        : [[dh-5,25,2], [dh-5,-35 ,2], [x1,y1,2], [x2,y2,2]],
        'RIS_elements'           : (8,8),
        'RIS_element_groups'     : (4,1),
        'RIS_phase_values'       : [np.exp(1j*0), np.exp(1j*1)],
        'TX_locations'           : [0,30,2],
        'TX_RIS_link_mult_factor': Power,
        'RX_RIS_link_mult_factor': Power,
        'TX_RX_link_mult_factor' : Power,
        'TX_RX_link_is_LOS'      : not isWall,
        'train_RX_square_center' : (20,30),
        'train_RX_square_width'  : 1,
        'train_RX_num_positions' : 100,
        'train_RX_height'        : 1,
        'noise_power'            : 100,
        }})

    RIS_list,\
    TX_RIS_link_info, \
    RX_RIS_link_info, \
    TX_RX_link_info, \
    train_RX_locations, \
    test_RX_locations = initialize_simulation_from_setup(setup)

    ch = Channel(setup.TX_locations, train_RX_locations[0,:], TX_RIS_link_info, RX_RIS_link_info, TX_RX_link_info, setup.noise_power)


    occurancies = dict()

    N = 10
    for _ in range(1):

        best_configuration, best_snr = find_RIS_configuration_that_maximizes_SNR(RIS_list, ch, show_progress_bar=False)

        if best_configuration in occurancies.keys():
            occurancies[best_configuration] += 1
        else:
            occurancies[best_configuration] = 0

    for key,value in occurancies.items():
        print("{} : {:.1f}%".format(key, 100*value/float(N)))


    chi_2, p_value = stats.chisquare(np.array(list(occurancies.values()))/float(N), [1/64.0]*len(occurancies.values()))
    print("χ^2: ", chi_2)
    print("p value: ", p_value)



