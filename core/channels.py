import numpy as np
from copy import deepcopy
from typing import *

from core import globals
from core.surfaces import RIS
from utils.misc import Vector, Matrix2D, convert2array, Vector3D
from utils.complex import sample_gaussian_complex_matrix


def calc_pathloss(tx_location,
                  rx_location,
                  isLOS,):

    myDist = np.sqrt(np.sum(np.power(tx_location-rx_location, 2)))
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

    def initialize(self, sender_coordinates: Matrix2D, receiver_coordinates: Matrix2D,):
        super().initialize(sender_coordinates, receiver_coordinates)

        self.fades               = sample_gaussian_complex_matrix((self.num_senders, self.num_receivers)) / np.sqrt(2)
        self.path_losses         = np.empty_like(self.fades)
        self.transmission_matrix = np.empty_like(self.fades)

        for i in range(self.num_senders):
            for j in range(self.num_receivers):
                self.path_losses[i,j] = calc_pathloss(self.sender_coordinates[i,:],
                                                      self.receiver_coordinates[j,:],
                                                      self.isLOS)
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



    def _calculate_SNR(self, H, Phi, G, h):
        channel_reflected = G @ Phi @ H + h # The @ operator denotes matrix multiplication (Python >= 3.5 - PEP 465)
        snr = np.power(np.absolute(channel_reflected), 2) / self.noise_power
        return float(snr)



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






