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

    # ris_elements     = globals.total_elements
    # groups_of        = globals.dependent_elements
    # number_surfaces  = globals.total_surfaces
    # element_size     = [globals.wavelength/2, globals.wavelength/2]
    # in_group_dist    = [globals.wavelength, globals.wavelength]
    # out_group_dist   = [globals.wavelength, globals.wavelength]
    # phase_space      = ('discrete', {'values': [np.exp(1j*0), np.exp(1j*np.pi)]})
    # ris_coords       = np.array([[10, 10, 2],])
    # tx_coords        = globals.TX_location
    # rx_coords        = globals.RX_location
    # mult_factor      = globals.mult_fact
    # tx_ris_link_info = (RayleighFadeLink, {'mult_factor': mult_factor, 'isLOS': True})
    # ris_rx_link_info = (RayleighFadeLink, {'mult_factor': mult_factor, 'isLOS': True})
    # tx_rx_link_info  = (RayleighFadeLink, {'mult_factor': mult_factor, 'isLOS': False})
    # noise_power      = globals.noisePower
    # r1               = RIS(ris_coords[0, :], ris_elements, groups_of, element_size, in_group_dist, out_group_dist, phase_space)
    #
    #
    # ris_list = [r1]
    #
    #
    # plot_positions(ris_coords, np.array([tx_coords]), np.array([rx_coords]))
    # plt.show()
    #
    # r1.set_random_state()

    setup = Setup.from_config({'test': {
        'num_RIS': 3,
        'RIS_coordinates': [[0,0,0],[1,1,1],[2,2,2]],
        'RIS_elements': (4,4),
        'RIS_element_groups': (2,2),
        'RIS_phase_values': [-1, 1],
        'TX_locations'    : [-1,-1,-1],
        'TX_RIS_link_mult_factor': 1,
        'RX_RIS_link_mult_factor': 1,
        'TX_RX_link_mult_factor': 1,
        'train_RX_square_center': (10,10),
        'train_RX_square_width' : 1,
        'train_RX_num_positions' : 100,
        'train_RX_height'        : 1,
        'noise_power'            : 1,
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
    for _ in range(N):

        best_configuration, best_snr = find_RIS_configuration_that_maximizes_SNR(RIS_list, ch)

        if best_configuration in occurancies.keys():
            occurancies[best_configuration] += 1
        else:
            occurancies[best_configuration] = 0

    for key,value in occurancies.items():
        print("{} : {:.1f}%".format(key, 100*value/float(N)))


    chi_2, p_value = stats.chisquare(np.array(list(occurancies.values()))/float(N), [1/64.0]*len(occurancies.values()))
    print("χ^2: ", chi_2)
    print("p value: ", p_value)



