import numpy as np



from core.surfaces import RIS
from core.experiment_setups import PositionGrid
from core.channels import Channel, GaussianFadeLink

if __name__ == '__main__':

    ris_elements     = 6
    groups_of        = 2
    number_surfaces  = 2
    element_size     = [0.1, 0.1]
    in_group_dist    = [0.1, 0.1]
    out_group_dist   = [0.2, 0.2]
    state_space      = ('binary', {})
    phase_space      = ('discrete', {'values': [np.exp(1j*0), np.exp(1j*np.pi)]})

    ris_coords       = np.array([[0, 0, 0],
                                 [1, 1, 1]])
    tx_coords        = np.array([[2, 2, 2]])
    rx_coords        = np.array([[3, 3, 3]])

    r1 = RIS(ris_coords[0, :], (ris_elements, 1), (groups_of, 1), element_size, in_group_dist, out_group_dist, state_space, phase_space)
    r2 = RIS(ris_coords[1, :], (ris_elements, 1), (groups_of, 1), element_size, in_group_dist, out_group_dist, state_space, phase_space)

    r1.set_random_state()
    r2.set_random_state()

    pg = PositionGrid(ris_coords, tx_coords, rx_coords)

    mult_factor = 1
    link_info = (GaussianFadeLink, {'mult_factor': mult_factor})
    noise_power = 1

    ch = Channel(pg, [r1, r2], ris_elements, link_info, link_info, link_info, noise_power)
    ch.simulate_transmission()
    ch.set_RIS_phases([r1, r2])

    for _ in range(20):
        r1.set_random_state()
        r2.set_random_state()
        ch.simulate_transmission()
        print("{:.4e}".format(ch.get_SNR()))