import numpy as np

from deprecated.setup import Setup, initialize_simulation_from_setup
from deprecated.channels import Channel
from deprecated import exhaustive_snr_search

from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':

    start_t = datetime.now()




    dh           = 20
    x1 = x2 = y1 = 25
    y2           = 35
    isWall       = True
    Power        = 1

    setup = Setup.from_config({'test': {
        'num_RIS'                : 4,
        'RIS_coordinates'        : [[dh-5,25,2], [dh-5,-35 ,2], [x1,y1,2], [x2,y2,2]],
        'RIS_elements'           : (8,8),
        'RIS_element_groups'     : (4,4),
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
    test_RX_locations, \
    center_TX_position = initialize_simulation_from_setup(setup)

    from utils.plotting import plot_setup_3D, plot_positions, grid_plot_params



    params = grid_plot_params.copy()
    params['zlims'] = [0, 3]
    params['color_by_height'] = False
    plot_setup_3D(RIS_list, setup.TX_locations.reshape((1,3)), center_TX_position.reshape(1,3), params=params)

    plot_positions(np.array([ris.position for ris in RIS_list]), setup.TX_locations.reshape((1, 3)), center_TX_position.reshape(1, 3),)

    plt.show()

    for i in [1]:#tqdm(range(train_RX_locations.shape[0])):

        ch = Channel(setup.TX_locations, train_RX_locations[i,:], TX_RIS_link_info, RX_RIS_link_info, TX_RX_link_info, setup.noise_power)
        best_configuation, best_snr, _, _, _, _, =  exhaustive_snr_search(ch, RIS_list, batch_size=2 ** 11, show_progress_bar=False) #ch.exhaustive_snr_search(RIS_list, batch_size=2 ** 11, show_progress_bar=False)













    end_t = datetime.now()
    duration = end_t-start_t
    print("Run simulation with {} RIS, {} phases, {} elements grouped in {}.  Time elapsed: {}".format(
        setup.num_RIS, len(setup.RIS_phase_values), setup.RIS_elements, setup.RIS_element_groups, duration))



