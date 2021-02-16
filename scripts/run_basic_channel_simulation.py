import numpy as np
from scipy import stats



from core.surfaces import RIS
from core.experiment_setups import PositionGrid
from core.channels import Channel, GaussianFadeLink
from core.exhaustive_phase_search import find_RIS_configuration_that_maximizes_SNR
import core.globals as globals

if __name__ == '__main__':

    ris_elements     = globals.total_elements
    groups_of        = globals.dependent_elements
    number_surfaces  = globals.total_surfaces
    element_size     = [globals.wavelength/2, globals.wavelength/2]
    in_group_dist    = [globals.wavelength, globals.wavelength]
    out_group_dist   = [globals.wavelength, globals.wavelength]
    phase_space      = ('discrete', {'values': [np.exp(1j*0), np.exp(1j*np.pi)]})

    ris_coords       = np.array([[0, 0, 0],
                                 [1, 1, 1]])
    tx_coords        = globals.TX_location
    rx_coords        = globals.RX_location

    r1 = RIS(ris_coords[0, :], (ris_elements, 1), (groups_of, 1), element_size, in_group_dist, out_group_dist, phase_space)
    r2 = RIS(ris_coords[1, :], (ris_elements, 1), (groups_of, 1), element_size, in_group_dist, out_group_dist, phase_space)

    r1.set_random_state()
    r2.set_random_state()

    #pg = PositionGrid(ris_coords, tx_coords, rx_coords)

    mult_factor = globals.mult_fact
    link_info = (GaussianFadeLink, {'mult_factor': mult_factor})
    noise_power = globals.noisePower

    ch = Channel(tx_coords, rx_coords, link_info, link_info, link_info, noise_power)

    for _ in range(20):
        r1.set_random_state()
        r2.set_random_state()
        snr = ch.simulate_transmission([r1,r2])
        print("{:.4e}".format(snr))

    occurancies = dict()

    N = 50
    for _ in range(N):

        best_configuration, best_snr = find_RIS_configuration_that_maximizes_SNR([r1,r2], ch)

        if best_configuration in occurancies.keys():
            occurancies[best_configuration] += 1
        else:
            occurancies[best_configuration] = 0

    for key,value in occurancies.items():
        print("{} : {:.1f}%".format(key, 100*value/float(N)))


    chi_2, p_value = stats.chisquare(np.array(list(occurancies.values()))/float(N), [1/64.0]*len(occurancies.values()))
    print("χ^2: ", chi_2)
    print("p value: ", p_value)



