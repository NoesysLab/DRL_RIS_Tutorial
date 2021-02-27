from typing import *

import numpy as np
from tqdm import tqdm

from core.surfaces import *
from core.channels import *
from core.geometry import *


def iterate_combined_state(ris_list: List[RIS]):






    return




def find_RIS_configuration_that_maximizes_SNR(ris_list: List[RIS], channel: Channel, show_progress_bar=False):

    def all_ris_have_same_number_or_elements(ris_list):
        return all([ris.num_tunable_elements == ris_list[0].num_tunable_elements and ris.total_elements == ris_list[0].total_elements for ris in ris_list])

    def are_all_states_of_equal_values(ris_list):
        return all([len(ris.phase_space.values)==len(ris_list[0].phase_space.values) for ris in ris_list])



    if not all_ris_have_same_number_or_elements(ris_list):
        raise ValueError("Currently supporting RISs of equal number of elements (tunable and total).")

    if not are_all_states_of_equal_values(ris_list):
        raise ValueError("All RISs must have the same number of discrete states.")


    total_tunable_elements        = sum([ris.num_tunable_elements for ris in ris_list])
    dependent_elements_per_RIS    = ris_list[0].num_dependent_elements
    discrete_states               = range(ris_list[0].state_space.num_values)
    phase_space                   = ris_list[0].phase_space
    combined_state_space_elements = int(len(discrete_states)**total_tunable_elements)

    possible_configurations       = itertools.product(discrete_states, repeat=total_tunable_elements)


    best_snr               = -float('inf')
    best_configuration     = None



    if show_progress_bar: possible_configurations = tqdm(possible_configurations, total=combined_state_space_elements)

    for configuration in possible_configurations:

        phase = phase_space.calculate_phase_shifts(configuration)
        phase = np.repeat(phase, repeats=dependent_elements_per_RIS)
        snr   = channel.simulate_transmission(ris_list, combined_RIS_phase=phase)[0]

        if snr > best_snr:
            best_snr = snr
            best_configuration = configuration

    return best_configuration, best_snr