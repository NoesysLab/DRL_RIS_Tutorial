import numpy as np
from typing import *

from tqdm import tqdm

from deprecated import globals
from core.surfaces import RIS
from utils.binary_space import BinaryEnumerator
from utils.misc import Vector, Matrix2D, Vector3D, sample_gaussian_complex_matrix


def calc_pathloss(tx_location,
                  rx_location,
                  isLOS,):

    myDist = np.sqrt(np.sum(np.power(tx_location-rx_location, 2), axis=1)).reshape((tx_location.shape[0], rx_location.shape[0]))
    if isLOS:
        pathlossDB = globals.pathlossCoefficientLOS \
                     + 10 * globals.pathlossExponentLOS \
                     * np.log10(myDist / globals.referenceDistance)
    else:
        pathlossDB = globals.pathlossCoefficientNLOS \
                     + 10 * globals.pathlossExponentNLOS \
                     * np.log10(myDist / globals.referenceDistance)

    myPthLoss = np.power(10, (pathlossDB/10))
    return myPthLoss







class Link:
    def __init__(self):
        self.sender_coordinates   = None
        self.receiver_coordinates = None
        self.num_senders          = None
        self.num_receivers        = None
        self.num_transmissions    = 1

    def initialize(self, sender_coordinates: Matrix2D, receiver_coordinates: Matrix2D):
        self.sender_coordinates   = np.array(sender_coordinates)
        self.receiver_coordinates = np.array(receiver_coordinates)
        self.num_senders          = self.sender_coordinates.shape[0]
        self.num_receivers        = self.receiver_coordinates.shape[0]

    def get_transmission_matrix(self)->Matrix2D:
        raise NotImplemented









class RayleighFadeLink(Link):
    def __init__(self,  mult_factor: float, isLOS=True):
        super().__init__()
        self.mult_factor         = mult_factor
        self.isLOS               = isLOS
        self.fades               = None
        self.path_losses         = None
        self.transmission_matrix = None

    def initialize(self, sender_coordinates: Matrix2D, receiver_coordinates: Matrix2D):
        super().initialize(sender_coordinates, receiver_coordinates)

        self.fades               = sample_gaussian_complex_matrix((self.num_senders, self.num_receivers)) / np.sqrt(2)
        self.path_losses         = np.empty_like(self.fades)
        self.transmission_matrix = np.empty_like(self.fades)

        # for i in range(self.num_senders):
        #     for j in range(self.num_receivers):
        #         self.path_losses[i,j] = calc_pathloss(self.sender_coordinates[i,:],
        #                                               self.receiver_coordinates[j,:],
        #                                               self.isLOS)
        self.path_losses = calc_pathloss(self.sender_coordinates, self.receiver_coordinates, self.isLOS)
        self.transmission_matrix = self.mult_factor * self.fades / np.sqrt(self.path_losses)


    def get_transmission_matrix(self)->Matrix2D:
        return np.transpose(self.transmission_matrix)




class Channel:
    def __init__(self,
                 TX_position          : Vector3D,
                 RX_position          : Vector3D,
                 TX_RIS_link_info     : Tuple[Type[Link], Dict],
                 RX_RIS_link_info     : Tuple[Type[Link], Dict],
                 TX_RX_link_info      : Tuple[Type[Link], Dict],
                 noise_power          : float):


        self.tx_position             = TX_position              # type: Vector3D
        self.rx_position             = RX_position              # type: Vector3D
        self.noise_power             = noise_power              # type: float

        link_type, link_args         = TX_RIS_link_info
        self.TX_RIS_link             = link_type(**link_args)   # type: Link

        link_type, link_args         = RX_RIS_link_info
        self.RX_RIS_link             = link_type(**link_args)   # type: Link

        link_type, link_args         = TX_RX_link_info
        self.TX_RX_link              = link_type(**link_args)   # type: Link

        self.ris_phases              = None                     # type: Vector # length: num_elements_per_ris * num_ris



    @staticmethod
    def get_combined_elements_coordinates(ris_list: List[RIS]):
        return np.concatenate([ris.get_element_coordinates() for ris in ris_list], axis=0)

    @staticmethod
    def get_combined_ris_phase(ris_list: List[RIS]):
        return np.concatenate([ris.get_phase('1D') for ris in ris_list])



    def _calculate_SNR(self, H, Phi, G, h=None):
        channel_reflected = G @ Phi @ H # The @ operator denotes matrix multiplication (Python >= 3.5 - PEP 465)
        if h is not None:
            channel_reflected += h
        snr = np.power(np.absolute(channel_reflected), 2) / self.noise_power
        return snr



    def simulate_transmission(self, ris_list: List[RIS], combined_RIS_phase=None):

        ris_element_coordinates = self.get_combined_elements_coordinates(ris_list)

        if combined_RIS_phase is not None:
            ris_phases = combined_RIS_phase
            if len(ris_phases) != sum([ris.total_elements for ris in ris_list]) :
                raise ValueError("Combined RIS phase does not equal to the total number of independent RIS elements.")
        else:
            ris_phases = self.get_combined_ris_phase(ris_list)

        self.TX_RIS_link.initialize([self.tx_position]     , ris_element_coordinates)
        self.RX_RIS_link.initialize(ris_element_coordinates, [self.rx_position])
        self.TX_RX_link.initialize([self.tx_position]      , [self.rx_position])

        H   = self.TX_RIS_link.get_transmission_matrix() # shape: (K,1)
        G   = self.RX_RIS_link.get_transmission_matrix() # shape: (1,K)
        h   = self.TX_RX_link.get_transmission_matrix()  # shape: (1,1)
        Phi = np.diag(ris_phases)                        # shape: (K,K)


        snr      = self._calculate_SNR (H, Phi, G, h)
        angle_TX = np.angle(H)
        mag_TX   = np.abs(H)
        angle_RX = np.angle(G)
        mag_RX   = np.abs(G)

        return snr, angle_TX, mag_TX, angle_RX, mag_RX


    @staticmethod
    def all_ris_have_same_number_or_elements(ris_list):
        return all([ris.num_tunable_elements == ris_list[0].num_tunable_elements and ris.total_elements == ris_list[0].total_elements for ris in ris_list])

    @staticmethod
    def are_all_states_of_equal_values(ris_list):
        return all([len(ris.phase_space.values)==len(ris_list[0].phase_space.values) for ris in ris_list])








    def exhaustive_snr_search(self, ris_list: List[RIS], batch_size=2**11, show_progress_bar=False):

        if not self.all_ris_have_same_number_or_elements(ris_list):
            raise ValueError("Currently supporting RISs of equal number of elements (tunable and total).")

        if not self.are_all_states_of_equal_values(ris_list):
            raise ValueError("All RISs must have the same number of discrete states.")




        total_tunable_elements        = sum([ris.num_tunable_elements for ris in ris_list])
        dependent_elements_per_RIS    = ris_list[0].num_dependent_elements
        discrete_states               = range(ris_list[0].state_space.num_values)
        phase_space                   = ris_list[0].phase_space
        combined_state_space_elements = int(len(discrete_states)) ** int(total_tunable_elements)



        num_transmissions             = combined_state_space_elements
        K                             = sum([ris.total_elements for ris in ris_list])

        ris_element_coordinates       = self.get_combined_elements_coordinates(ris_list)

        self.TX_RIS_link.initialize([self.tx_position], ris_element_coordinates)
        self.RX_RIS_link.initialize(ris_element_coordinates, [self.rx_position])
        self.TX_RX_link.initialize([self.tx_position], [self.rx_position])


        num_batches_required = int(np.ceil(num_transmissions / batch_size))
        last_batch_size      = batch_size if num_transmissions % batch_size == 0 else num_transmissions % batch_size

        #possible_configurations = itertools.product(discrete_states, repeat=total_tunable_elements)
        possible_configurations = BinaryEnumerator(batch_size, total_tunable_elements)

        best_batch_results = []
        best_batch_snrs    = np.empty(shape=(num_batches_required,))


        batch_indices = range(num_batches_required)
        if show_progress_bar: batch_indices = tqdm(batch_indices, leave=True)

        for i in tqdm(batch_indices):

            batch_transmissions  = batch_size if i != num_batches_required-1 else last_batch_size

            #batch_configurations = [next(possible_configurations) for _ in range(batch_transmissions)]
            batch_configurations = next(possible_configurations)

            batch_phases = phase_space.calculate_phase_shifts(batch_configurations)
            batch_phases = np.repeat(batch_phases, repeats=dependent_elements_per_RIS, axis=1)

            # batch_phases = []
            # for configuration in batch_configurations:
            #     phase = phase_space.calculate_phase_shifts(configuration)
            #     phase = np.repeat(phase, repeats=dependent_elements_per_RIS)
            #     batch_phases.append(phase)
            #
            # batch_phases = np.array(batch_phases)


            try:
                H   = np.empty(shape=(batch_transmissions, K, 1), dtype=np.complex)
                G   = np.empty(shape=(batch_transmissions, 1, K), dtype=np.complex)
                h   = np.empty(shape=(batch_transmissions, 1, 1), dtype=np.complex)
                Phi = np.empty(shape=(batch_transmissions, K, K), dtype=np.complex)

            except MemoryError as e:
                required_memory = str(e).split("Unable to allocate ")[-1].split(" for an array")[0]
                raise MemoryError("Each batch requires "+required_memory+" which exceeds system memory. Lower batch_size value and try again.")

            for j in range(batch_transmissions):

                #transmission_index = i*batch_size+j

                H[j,:,:]   = self.TX_RIS_link.get_transmission_matrix()  # shape: (K,1)
                G[j,:,:]   = self.RX_RIS_link.get_transmission_matrix()  # shape: (1,K)
                h[j,:,:]   = self.TX_RX_link.get_transmission_matrix()   # shape: (1,1)
                Phi[j,:,:] = np.diag(batch_phases[j,:])      # shape: (K,K)


            all_snr = self._calculate_SNR(H, Phi, G, h).flatten() # type: np.ndarray
            #all_snr = self._calculate_SNR(H, Phi, G).flatten()  # type: np.ndarray

            best_configuration_index = np.argmax(all_snr)
            snr                      = all_snr[best_configuration_index]

            best_configuration       = batch_configurations[best_configuration_index]
            angle_TX                 = np.angle( H[best_configuration_index, :, :])
            mag_TX                   = np.abs(   H[best_configuration_index, :, :])
            angle_RX                 = np.angle( G[best_configuration_index, :, :])
            mag_RX                   = np.abs(   G[best_configuration_index, :, :])


            best_batch_results.append((best_configuration, snr, angle_TX, mag_TX, angle_RX, mag_RX, best_configuration_index))
            best_batch_snrs[i] = snr


        best_snr_index_from_batches = int(np.argmax(best_batch_snrs))
        best_configuration, snr, angle_TX, mag_TX, angle_RX, mag_RX, best_configuration_index = best_batch_results[best_snr_index_from_batches]

        return best_configuration, snr, angle_TX, mag_TX, angle_RX, mag_RX,
