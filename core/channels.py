import numpy as np
from copy import deepcopy
from typing import *

from core import globals
from core.surfaces import RIS
from core.experiment_setups import PositionGrid
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
    def __init__(self, sender_coordinates: Matrix2D, receiver_coordinates: Matrix2D):
        self.sender_coordinates   = sender_coordinates
        self.receiver_coordinates = receiver_coordinates
        self.num_senders          = self.sender_coordinates.shape[0]
        self.num_receivers        = self.receiver_coordinates.shape[0]


    def initialize(self):
        raise NotImplemented

    def get_transmission_matrix(self)->Matrix2D:
        raise NotImplemented









class GaussianFadeLink(Link):
    def __init__(self,  sender_coordinates: Matrix2D, receiver_coordinates: Matrix2D, mult_factor: float, isLOS=True):
        super().__init__(sender_coordinates, receiver_coordinates)
        self.mult_factor         = mult_factor
        self.isLOS               = isLOS
        self.fades               = np.empty((self.num_senders, self.num_receivers))
        self.path_losses         = np.empty_like(self.fades)
        self.transmission_matrix = np.empty_like(self.fades)


    def initialize(self):
        self.fades       = sample_gaussian_complex_matrix(self.fades.shape) / np.sqrt(2)
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
                 position_grid        : PositionGrid,
                 ris_list             : List[RIS],
                 num_elements_per_ris : int,
                 TX_RIS_link_info     : Tuple[Type[Link], Dict],
                 RX_RIS_link_info     : Tuple[Type[Link], Dict],
                 TX_RX_link_info      : Tuple[Type[Link], Dict],
                 noise_power          : float):

        assert position_grid.num_RX == 1, 'Currently only supporting channel of a single RX.'
        assert position_grid.num_TX == 1, 'Currently only supporting channel of a single TX.'

        self.position_grid           = position_grid
        self.noise_power             = noise_power
        self.ris_list                = ris_list
        self.num_elements_per_ris    = num_elements_per_ris
        self.all_ris_elements_coords = self.get_combined_elements_coordinates(self.ris_list)

        link_type, link_args         = TX_RIS_link_info
        self.TX_RIS_link             = link_type(self.position_grid.TX_positions, self.all_ris_elements_coords, **link_args)

        link_type, link_args         = RX_RIS_link_info
        self.RX_RIS_link             = link_type(self.all_ris_elements_coords, self.position_grid.RX_positions, **link_args)

        link_type, link_args         = TX_RX_link_info
        self.TX_RX_link              = link_type(self.position_grid.TX_positions, self.position_grid.RX_positions, **link_args)


        self.ris_phases              = None             # type: Vector # length: num_elements_per_ris * num_ris



    @staticmethod
    def get_combined_elements_coordinates(ris_list: List[RIS]):
        return np.concatenate([ris.get_element_coordinates() for ris in ris_list], axis=0)



    @property
    def H(self)->Matrix2D: # shape: (K,1)
        H = self.TX_RIS_link.get_transmission_matrix()
        return H
    @property
    def G(self)->Matrix2D: # shape (1,K)
        G =  self.RX_RIS_link.get_transmission_matrix()
        return G

    @property
    def h(self)->Vector:  # shape: (1,1)
        h = self.TX_RX_link.get_transmission_matrix()
        return h

    @property
    def Phi(self)->Matrix2D: # shape: (K,K)
        Phi = np.diag(self.ris_phases)
        return Phi



    def set_RIS_phases(self, ris_list: List[RIS]):
        def are_all_phases_of_equal_length(ris_phases: List[Vector]):
            return all(len(i) == len(ris_phases[0]) for i in ris_phases)

        ris_phases_1D = [ris.get_phase('1D') for ris in ris_list]
        assert are_all_phases_of_equal_length(ris_phases_1D), 'Currently only supporting RISs with phases of equal length.'

        self.ris_phases = np.concatenate(ris_phases_1D)



    def simulate_transmission(self):
        self.TX_RIS_link.initialize()
        self.RX_RIS_link.initialize()
        self.TX_RX_link.initialize()


    def get_SNR(self):
        channel_reflected = self.G @ self.Phi @ self.H + self.h # The @ operator denotes matrix multiplication (Python >= 3.5 - PEP 465)
        # todo: PL_TX_RX is not modelled here

        snr = np.power(np.absolute(channel_reflected), 2) / self.noise_power
        return float(snr)








