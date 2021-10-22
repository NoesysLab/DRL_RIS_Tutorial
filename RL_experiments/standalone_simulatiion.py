import sys

import numpy as np
from scipy.constants import pi, speed_of_light
from dataclasses import dataclass, field
import json

from tqdm import tqdm

dcArray = lambda X: field(default=np.array(X))

SPEED_OF_LIGHT_AIR = 299702547.0


################################# Configuration and Default Parameters #################################################

@dataclass()
class Setup:
    """
    Parameters of the Setup to be used.
    Attributes with no default values must be set explicitly whc constructing the class
    Attributes set to value `field(init=False)` will have their value set in __post__init__() method using values from other parameters
    An instance of this class can be created using the `load_from_json()` method from a JSON file that contains the parameter values.
    """

    K                          : int                                       # Number of users
    B                          : int                                       # Number of BS antennas
    M                          : int                                       # Number of RIS
    N                          : int                                       # Number of RIS elements
    transmit_power             : float                                     # (dBm)
    noise_variance             : float                                     # Variance of the AWGN - assumed equal for all RXs (dBm)
    codebook_rays_per_RX       : int
    group_size                 : int                                       # Size of RIS element groups that share the same configuration. Must be a divisor of N.
    N_controllable             : int                                       # Total number of RIS element groups, calculated as N_tot / group_size. This can be thought of as the number of individually controlled elements.
    kappa_H                    : float                                     # BS-RIS Ricean factor (dB)
    kappa_G                    : float                                     # RIS-RX Ricean factor (dB)
    BS_position                : np.ndarray = dcArray([10, 5 , 2.0])
    RIS_positions              : np.ndarray = dcArray([[7.5, 13, 2.0], [12.5,  13,    2.0]])
    RX_box                     : np.ndarray = dcArray([ [7.5, 11.0, 1.5],  [12.5,  16.0, 2.0] ])
    direct_link_attenuation    : float      = None
    observation_noise_variance : float      = 0.
    ignore_direct_link         : bool       = False
    RIS_phases                 : np.ndarray = dcArray([0, pi])
    elem_dist                  : float      = field(default='wavelength')  # distance between elements in RIS
    RIS_resolution             : int        = 2
    frequency                  : float      = 5*10**9                      # Operating frequency (Hz)
    wavelength                 : float      = field(init=False)            # wavelength (m)
    num_RIS_phases             : int        = field(init=False)
    codebook                   : np.ndarray = field(init=False)            # Codebook of precoding matrices. Shape: (rays_per_user, B, K)
    N_tot                      : int        = field(init=False)            # Total number of RIS elements
    RX_positions               : np.ndarray = field(init=False)
    d_BS_RIS                   : np.ndarray = field(init=False)            # Shape: (M,)
    d_RIS_RX                   : np.ndarray = field(init=False)            # Shape: (M, K)
    d_BS_RX                    : np.ndarray = field(init=False)            # Shape: (K)



    def __post_init__(self):
        """
        Initialization of Setup's parameters that depend on other user-defined values
        """
        if self.N % self.group_size != 0:
            raise ValueError('Size of RIS element groups must divide perfectly the number of elements in a RIS.')

        self.N_tot          = self.N * self.M

        self.wavelength     = speed_of_light / self.frequency
        self.num_RIS_phases = 2 ** self.RIS_resolution
        self.elem_dist      = self.wavelength if self.elem_dist == 'wavelength' else self.elem_dist
        self.RX_positions   = np.array([[ 8.77504286, 14.39457326,  1.63428017], [ 9.64815658, 13.28167566,  1.63265777]])#get_random_RX_positions(self.K, self.RX_box[0, :], self.RX_box[1, :])
        self.d_BS_RIS       = calculate_distances(self.BS_position,   self.RIS_positions).flatten()
        self.d_RIS_RX       = calculate_distances(self.RIS_positions, self.RX_positions)  
        self.d_BS_RX        = calculate_distances(self.BS_position,   self.RX_positions).flatten()
        self.kappa_H        = dBW_to_Watt(self.kappa_H)
        self.kappa_G        = dBW_to_Watt(self.kappa_G)
        self.transmit_power = dBm_to_Watt(self.transmit_power)
        self.noise_variance = dBm_to_Watt(self.noise_variance)
        self.BS_position    = np.array(self.BS_position)
        self.RIS_positions  = np.array(self.RIS_positions)
        self.RX_box         = np.array(self.RX_box)

        # if self.direct_link_attenuation is not None:
        #     self.direct_link_attenuation = dBW_to_Watt(-1 * self.direct_link_attenuation)
        self.codebook      = initialize_precoding_codebook(self)

        if self.N_controllable is None:
            self.N_controllable = self.N_tot // self.group_size
        else:
            assert self.N_controllable == self.N_tot // self.group_size

        #print(f'RXs are positioned at:\n{self.RX_positions}')


    @staticmethod
    def load_from_json(file_or_filename: str):
        if isinstance(file_or_filename, str):
            file = open(file_or_filename, 'r')
        else:
            file = file_or_filename

        setup_params = json.loads(file.read())
        try:
            setup = Setup(**setup_params['SETUP'])
            file.close()
            return setup
        except TypeError as e:
            file.close()
            raise ValueError(f"Unable to initialize setup from JSON file. Error:\n{e}")




################################## Helper Functions ####################################################################


def dBm_to_Watt(val_dBm):
    return np.power(10, (val_dBm/10 - 3)  )

def dBW_to_Watt(val_dBW):
    return np.power(10, val_dBW/10)


def get_random_RX_positions(num_RXs, box_low, box_high):
    positions = np.empty((num_RXs, 3))
    for i in range(num_RXs):
        positions[i,:] = np.random.uniform(low=box_low, high=box_high)
    return positions

def calculate_distances(A: np.ndarray, B: np.ndarray)->np.ndarray:

    if A.ndim == 1: A = A.reshape((1,-1))
    if B.ndim == 1: B = B.reshape((1,-1))

    assert A.shape[1] == B.shape[1] == 3
    n     = A.shape[0]
    m     = B.shape[0]
    dists = np.empty((n,m))
    for i in range(n):
        a = A[i,:]
        dists[i,:] = np.linalg.norm(a - B)

    return dists

def sample_gaussian_standard_normal(size=None):
    betta = np.random.normal(0, 1, size=size) + 1j * np.random.normal(0, 1, size=size)
    return betta


def split_to_close_to_square_factors(x: int):
    n1 = int(np.floor(np.sqrt(x)))
    while n1 >= 1:
        if x % n1 == 0: break
        n1 -= 1
    n2 = x // n1
    return n1, n2

def ray_to_elevation_azimuth(starting_point, ending_point):
    def cart2sph(x, y, z):
        XsqPlusYsq = x ** 2 + y ** 2
        r = np.sqrt(XsqPlusYsq + z ** 2)  # r
        elev = np.arctan2(z, np.sqrt(XsqPlusYsq))  # theta
        az = np.arctan2(y, x)  # phi
        return r, elev, az

    v = ending_point - starting_point # type: np.ndarray
    _, elev, az = cart2sph(v[0], v[1], v[2])
    return elev, az

############################################ Channel Equations #########################################################


def calculate_pathloss(setup: Setup, dist, extra_attenuation_dB=None):
    pl_dB = -20 * np.log10( 4 * pi * dist / setup.wavelength )
    if extra_attenuation_dB is not None:
        pl_dB -= extra_attenuation_dB
    pl_W = dBW_to_Watt(pl_dB)
    return pl_W

# def calculate_array_response(setup: Setup, num_RIS_elements: int, RIS_position: np.ndarray, BS_or_RX_position: np.ndarray):
#     """
#     ULA model
#     """
#     dist      = calculate_distances(RIS_position, BS_or_RX_position)
#     cos_angle = (RIS_position[0] - BS_or_RX_position[0]) / dist
#     n         = np.arange(num_RIS_elements)
#     a         = np.exp( (-1j * 2 * pi / setup.wavelength) * n * setup.elem_dist * cos_angle)
#
#     assert a.shape[0] == setup.N
#     return a


def calculate_array_response(setup: Setup, num_RIS_elements: int, RIS_position: np.ndarray, BS_or_RX_position: np.ndarray, direction_wrt_RIS=''):
    """
    from [Basar 2020]: "Indoor and Outdoor Physical Channel Modeling and Efficient Positioning for Reconfigurable Intelligent Surfaces in mmWave Bands"
    """
    if direction_wrt_RIS == 'arrival':
        starting_point = BS_or_RX_position
        ending_point   = RIS_position
    elif direction_wrt_RIS == 'departure':
        starting_point = RIS_position
        ending_point   = BS_or_RX_position
    else:
        raise ValueError("Expected one of 'arrival', 'departure' as direction.")

    d             = setup.elem_dist
    k             = 2 * pi / setup.wavelength
    theta, phi    = ray_to_elevation_azimuth(starting_point, ending_point)

    N_vert, N_hor = split_to_close_to_square_factors(num_RIS_elements)
    coords        = np.array([ (x,y) for x in range(N_hor) for y in range(N_vert) ])
    x             = coords[:,0]
    y             = coords[:,1]
    a             = np.exp(1j * k * d * (x * np.sin(theta) + y * np.sin(phi) * np.cos(theta)) )
    a             = a/np.linalg.norm(np.absolute(a))

    return a



def calculate_H(setup: Setup, BS_position, RIS_position, dist_BS_RIS):
    pl                  = calculate_pathloss(setup, dist_BS_RIS)
    RIS_steering_vector = calculate_array_response(setup, setup.N, RIS_position, BS_position, direction_wrt_RIS='arrival')
    BS_steering_vector  = calculate_array_response(setup, setup.B, setup.BS_position, RIS_position, direction_wrt_RIS='departure')
    a                   = BS_steering_vector.reshape((setup.B, 1)) @ RIS_steering_vector.reshape((1, setup.N))
    LOS_component       = np.sqrt(setup.kappa_H/(setup.kappa_H+1)) * a
    NLOS_component      = np.sqrt(1/(setup.kappa_H+1)) * sample_gaussian_standard_normal(size=(setup.B, setup.N))
    H                   = np.sqrt(pl) * ( LOS_component + NLOS_component )

    assert H.shape[0] == setup.B and H.shape[1] == setup.N
    return H


def calculate_G(setup: Setup, RIS_position, RX_position, dist_RIS_RX):
    pl             = calculate_pathloss(setup, dist_RIS_RX)
    LOS_component  = np.sqrt(setup.kappa_G / (setup.kappa_G + 1)) * calculate_array_response(setup, setup.N, RIS_position, RX_position, direction_wrt_RIS='departure')
    NLOS_component = np.sqrt(1 / (setup.kappa_G + 1)) * sample_gaussian_standard_normal(size=setup.N)
    G              = np.sqrt(pl) * (LOS_component + NLOS_component)

    assert G.shape[0] == setup.N
    return G

def calculate_h(setup : Setup, dist_TX_RX):
    pathloss = calculate_pathloss(setup, dist_TX_RX, extra_attenuation_dB=setup.direct_link_attenuation)

    h =  np.sqrt(pathloss) * sample_gaussian_standard_normal(size=setup.B)
    assert h.shape[0] == setup.B

    if setup.ignore_direct_link:
        h = np.zeros(shape=(setup.B,), dtype=complex)

    return h


def simulate_transmission(setup):

    H = np.empty(shape=(setup.M, setup.B, setup.N), dtype=complex)
    G = np.empty(shape=(setup.M, setup.K, setup.N), dtype=complex)
    h = np.empty(shape=(setup.K, setup.B), dtype=complex)


    for m in range(setup.M):
        H[m,:,:] = calculate_H(setup, setup.BS_position, setup.RIS_positions[m,:], setup.d_BS_RIS[m])

    for k in range(setup.K):
        h[k, :] = calculate_h(setup, setup.d_BS_RX[k])

        for m in range(setup.M):
            G[m,k,:] = calculate_G(setup, setup.RIS_positions[m,:], setup.RX_positions[k,:], setup.d_RIS_RX[m,k])

    return H, G, h


def compute_SINR_per_user(setup: Setup,
                          H,
                          G,
                          h,
                          RIS_profiles,
                          precoding_matrix):
    """
    Compute the SNR observed at every receiver

    :param setup
    :param H: (M, B, N)
    :param G: (M, K, N)
    :param h: (K, B)
    :param RIS_profiles: (M, N)
    :param precoding_matrix: (B, K)
    :return: (K,)
    """


    signal_strenghts = np.zeros(setup.K, dtype=np.float32)
    Phi              = np.zeros((setup.M, setup.N, setup.N), dtype=complex)

    for m in range(setup.M):
        Phi[m, :, :] = np.diag(RIS_profiles[m,:])

    for k in range(setup.K):
        total_RIS_cascaded_channel = np.zeros(setup.B, dtype=complex)
        for m in range(setup.M):
            H_m   = H[m,:,:]    # (B, N)
            Phi_m = Phi[m,:,:]  # (N, N)
            G_m   = G[m,k,:]    # (N, )
            G_m   = G_m.reshape((setup.N, 1))
            this_RIS_cascaded_channel =  H_m @ Phi_m @ G_m
            total_RIS_cascaded_channel += this_RIS_cascaded_channel.flatten()


        signal_strength = np.dot(h[k] + total_RIS_cascaded_channel, precoding_matrix[:,k])
        signal_strength = np.absolute(signal_strength) ** 2
        signal_strength = signal_strength.real

        signal_strenghts[k] = signal_strength

    sum_signal_strengths = np.sum(signal_strenghts)
    SINR = np.zeros(setup.K)
    for k in range(setup.K):
        SINR[k] = signal_strenghts[k] / (sum_signal_strengths - signal_strenghts[k]  + (setup.K * setup.noise_variance/ setup.transmit_power) )

    return SINR


def compute_sum_rate(SINR):
    return np.sum( np.log2(1 + SINR) )


def RIS_state2profile(theta):
    return np.exp(-1j * pi * theta)



#  # # # # # # # # # # # # # # Precoding matrix # # # # # # # #  # # # # # # # #

# def ULA_steering_vector(N, lam, theta, element_spacing):
#     n = np.arange(N)
#     return np.exp(-1j * 2 * pi * n * element_spacing * np.cos(theta) / lam )
#



def construct_precoding_matrix(setup: Setup, steering_vector_positions: np.ndarray) -> np.ndarray:
    """
    Given an array of L positions, a precoding matrix of shape (L,B) will be produced
    that contains at each l row the ULA steering vector to the l position
    :param setup: the Simulation setup. wavelength and element spacing for ULA will be set equivalent to the ones used for the RIS profile.
    :param steering_vector_positions: An array of coordinates of shape (L,3)
    :return:
    """
    assert steering_vector_positions.shape[1] == 3

    L = steering_vector_positions.shape[0]

    W = np.empty((setup.B, L), dtype=complex)
    for l in range(L):
        W[:,l]   = calculate_array_response(setup, setup.B, setup.BS_position, steering_vector_positions[l, :], 'departure')

    return W


def initialize_precoding_codebook(setup):

    codebook = np.empty((setup.codebook_rays_per_RX, setup.B, setup.K), dtype=complex)

    for i in range(setup.codebook_rays_per_RX):
        steering_positions = np.empty((setup.K, 3))
        for k in range(setup.K):
            steering_positions[k,:] = np.random.uniform(low=setup.RX_box[0,:], high=setup.RX_box[1,:])

        W = construct_precoding_matrix(setup, steering_positions)
        codebook[i,:,:] = W

    return codebook




def example_main():
    setup        = Setup.load_from_json("./parameters.json")                                     # Load setup parameters from configuration file

    avg_sum_rate = 0.0
    RUNS         = 1000
    for run in tqdm(range(RUNS)):
        thetas       = np.random.choice(setup.RIS_phases, size=(setup.M, setup.N))                   # Create a random RIS state (binary vector)
        RIS_profiles = RIS_state2profile(thetas)                                                     # Convert it to proper RIS profile
        H, G, h      = simulate_transmission(setup)                                                  # Sample channel realizations
        W            = setup.codebook[0,:,:]                                                         # Set the precoding matrix to be the first ray for each RX
        SINRs        = compute_SINR_per_user(setup, H, G, h, RIS_profiles, W)
        sum_rate     = compute_sum_rate(SINRs)
        avg_sum_rate+= sum_rate

    avg_sum_rate /= RUNS
    print(f"Average sum rate: {avg_sum_rate}")


if __name__ == '__main__':
    example_main()



