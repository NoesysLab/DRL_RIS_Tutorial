from typing import *

import numpy as np
from tqdm import tqdm

from core.surfaces import *
from core.channels import *
from core.geometry import *


def iterate_combined_state(ris_list: List[RIS]):

    def are_all_states_of_equal_values(ris_list):
        return all([len(ris.phase_space.values)==len(ris_list[0].phase_space.values) for ris in ris_list])

    if not are_all_states_of_equal_values(ris_list):
        raise ValueError("All RISs must have the same number of discrete states.")

    total_tunable_elements = sum([ris.num_tunable_elements for ris in ris_list])
    discrete_states        = range(ris_list[0].state_space.num_values)
    return itertools.product(discrete_states, repeat=total_tunable_elements)




def find_RIS_configuration_that_maximizes_SNR(ris_list: List[RIS], channel: Channel):

    def all_ris_have_same_number_or_elements(ris_list):
        return all([ris.num_tunable_elements == ris_list[0].num_tunable_elements and ris.total_elements == ris_list[0].total_elements for ris in ris_list])

    if not all_ris_have_same_number_or_elements(ris_list):
        raise ValueError("Currently supporting RISs of equal number of elements (tunable and total).")


    phase_space            = ris_list[0].phase_space
    num_dependent_elements = ris_list[0].total_elements / ris_list[0].num_tunable_elements

    best_snr           = -float('inf')
    best_configuration = None




    for configuration in iterate_combined_state(ris_list):

        phase = phase_space.calculate_phase_shifts(configuration)
        phase = np.repeat(phase, repeats=num_dependent_elements)
        snr   = channel.simulate_transmission(ris_list, combined_RIS_phase=phase)

        if snr > best_snr:
            best_snr = snr
            best_configuration = configuration

    return best_configuration, best_snr