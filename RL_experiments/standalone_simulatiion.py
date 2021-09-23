import numpy as np
from scipy.constants import pi, speed_of_light
from dataclasses import dataclass, field


dcArray = lambda X: field(default=np.array(X))

SPEED_OF_LIGHT_AIR = 299702547.0


################################# Configuration and Default Parameters #################################################

@dataclass()
class Setup:
    """
    Parameters of the Setup to be used.
    """

    K                          : int        = 3                                               # Number of users
    S                          : int        = 10                                              # Number of BS antennas
    M                          : int        = 2                                               # Number of RIS
    N                          : int        = 32                                               # Number of RIS elements
    N_tot                      : int        = field(init=False)                               # Total number of RIS elements
    group_size                 : int        = 4                                              # Size of RIS element groups that share the same configuration. Must be a divisor of N.
    N_controllable             : int        = field(init=False)                               # Total number of RIS element groups, calculated as N_tot / group_size. This can be thought of as the number of individually controlled elements.
    kappa_H                    : float      = 5.                                              # BS-RIS Ricean factor (dB)
    kappa_G                    : float      = 13.                                             # RIS-RX Ricean factor (dB)
    frequency                  : float      = 32*10**9                                        # Operating frequency (Hz)
    wavelength                 : float      = field(init=False)                               # wavelength (m)
    elem_dist                  : float      = field(default='wavelength')                     # distance between elements in RIS
    RIS_resolution             : int        = 2
    num_RIS_phases             : int        = field(init=False)
    RIS_phases                 : np.ndarray = dcArray([0, pi])
    precoding_v                : np.ndarray = field(default='ones')
    noise_variance             : float      = 100.                                             # Variance of the AWGN - assumed equal for all RXs (dBm)
    transmit_power             : float      = 1                                                # (Watt)
    BS_position                : np.ndarray = dcArray([10, 0 , 2.])
    RIS_positions              : np.ndarray = dcArray([[5, 25, 2.], [15, 25, 2.]])
    RX_box                     : np.ndarray = dcArray([[5., 20., 1.], [15., 30., 2.]])
    RX_positions               : np.ndarray = field(init=False)
    d_BS_RIS                   : np.ndarray = field(init=False)                               # Shape: (M,)
    d_RIS_RX                   : np.ndarray = field(init=False)                               # Shape: (M, K)
    d_BS_RX                    : np.ndarray = field(init=False)                               # Shape: (K)
    direct_link_attenuation    : float = None
    observation_noise_variance : float = 0.


    def __post_init__(self):
        """
        Initialization of Setup's parameters that depend on other user-defined values
        """
        if self.N % self.group_size != 0:
            raise ValueError('Size of RIS element groups must divide perfectly the number of elements in a RIS.')

        self.N_tot          = self.N * self.M
        self.N_controllable = self.N_tot // self.group_size
        self.wavelength     = speed_of_light / self.frequency
        self.num_RIS_phases = 2 ** self.RIS_resolution
        self.elem_dist      = self.wavelength if self.elem_dist == 'wavelength' else self.elem_dist
        self.precoding_v    = np.ones(self.S) if self.precoding_v == 'ones' else self.precoding_v

        self.precoding_v    = 1/np.linalg.norm(self.precoding_v)*self.precoding_v


        self.RX_positions   = get_random_RX_positions(self.K, self.RX_box[0, :], self.RX_box[1, :])
        self.d_BS_RIS       = calculate_distances(self.BS_position,   self.RIS_positions).flatten()
        self.d_RIS_RX       = calculate_distances(self.RIS_positions, self.RX_positions)  
        self.d_BS_RX        = calculate_distances(self.BS_position,   self.RX_positions).flatten()
        
        self.kappa_H        = dBW_to_Watt(self.kappa_H)
        self.kappa_G        = dBW_to_Watt(self.kappa_G)
        #self.transmit_power   = dBW_to_Watt(self.transmit_power)
        self.noise_variance = dBm_to_Watt(self.noise_variance)

        # if self.direct_link_attenuation is not None:
        #     self.direct_link_attenuation = dBW_to_Watt(-1 * self.direct_link_attenuation)



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

    both_are_1D = (A.ndim == 1 and B.ndim == 1)

    if A.ndim == 1: A = A.reshape((1,-1))
    if B.ndim == 1: B = B.reshape((1,-1))

    assert A.shape[1] == B.shape[1] == 3
    n     = A.shape[0]
    m     = B.shape[0]
    dists = np.empty((n,m))
    for i in range(n):
        a = A[i,:]
        dists[i,:] = np.linalg.norm(a - B)

    # if both_are_1D:
    #     return np.squeeze(dists, axis=1)
    # else:
    #     return np.squeeze(dists, axis=1)
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

    return a



def calculate_H(setup: Setup, BS_position, RIS_position, dist_BS_RIS):
    pl             = calculate_pathloss(setup, dist_BS_RIS)
    LOS_component  = np.sqrt(setup.kappa_H/(setup.kappa_H+1)) * calculate_array_response(setup, setup.N, RIS_position, BS_position, direction_wrt_RIS='arrival')
    LOS_component  = np.tile(LOS_component, (setup.S, 1))                         # Make array to (S,N) to account for multiple antennas
    NLOS_component = np.sqrt(1/(setup.kappa_H+1)) * sample_gaussian_standard_normal(size=(setup.S, setup.N))
    H              = np.sqrt(pl) * ( LOS_component + NLOS_component )

    assert H.shape[0] == setup.S and H.shape[1] == setup.N
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

    h =  np.sqrt(pathloss) * sample_gaussian_standard_normal(size=setup.S)
    assert h.shape[0] == setup.S
    return h


def simulate_transmission(setup):

    H = np.empty(shape=(setup.M, setup.S, setup.N), dtype=complex)
    G = np.empty(shape=(setup.M, setup.K, setup.N), dtype=complex)
    h = np.empty(shape=(setup.K, setup.S),          dtype=complex)


    for m in range(setup.M):
        H[m,:,:] = calculate_H(setup, setup.BS_position, setup.RIS_positions[m,:], setup.d_BS_RIS[m])

    for k in range(setup.K):
        h[k, :] = calculate_h(setup, setup.d_BS_RX[k])

        for m in range(setup.M):
            G[m,k,:] = calculate_G(setup, setup.RIS_positions[m,:], setup.RX_positions[k,:], setup.d_RIS_RX[m,k])

    return H, G, h


def compute_SINR_per_user(setup: Setup,
                          noise_variance,
                          H,
                          G,
                          h,
                          RIS_profiles,
                          precoding_vector):
    """
    Compute the SNR observed at every receiver

    :param setup
    :param noise_variance: In watt
    :param H: (M, S, N)
    :param G: (M, K, N)
    :param h: (K, S)
    :param RIS_profiles: (M, N)
    :param precoding_vector: (N,)
    :return: (K,)
    """


    signal_strenghts = np.zeros(setup.K, dtype=np.float32)
    Phi              = np.zeros((setup.M, setup.N, setup.N), dtype=complex)

    for m in range(setup.M):
        Phi[m, :, :] = np.diag(RIS_profiles[m,:])

    for k in range(setup.K):
        total_RIS_cascaded_channel = np.zeros(setup.S, dtype=complex)
        for m in range(setup.M):
            H_m   = H[m,:,:]    # (S, N)
            Phi_m = Phi[m,:,:]  # (N, N)
            G_m   = G[m,k,:]    # (N, )
            G_m   = G_m.reshape((setup.N, 1))
            this_RIS_cascaded_channel =  H_m @ Phi_m @ G_m
            total_RIS_cascaded_channel += this_RIS_cascaded_channel.flatten()


        signal_strength = np.dot(h[k] + total_RIS_cascaded_channel, precoding_vector)
        signal_strength = np.absolute(signal_strength) ** 2
        signal_strength = signal_strength.real

        signal_strenghts[k] = signal_strength

    sum_signal_strengths = np.sum(signal_strenghts)
    SINR = np.zeros(setup.K)
    for k in range(setup.K):
        SINR[k] = signal_strenghts[k] / (noise_variance + sum_signal_strengths - signal_strenghts[k])

    return SINR


def compute_sum_rate(SINR):
    return np.sum( np.log2(1 + SINR) )


def RIS_state2profile(theta):
    return np.exp(-1j * pi * theta)








def _main():
    setup        = Setup()

    thetas       = np.random.choice(setup.RIS_phases, size=(setup.M, setup.N))
    RIS_profiles = RIS_state2profile(thetas)
    H, G, h      = simulate_transmission(setup)
    SINR         = compute_SINR_per_user(setup, setup.noise_variance, H, G, h, RIS_profiles, setup.precoding_v)
    sum_rate     = compute_sum_rate(SINR)

    print(f"Sum rate: {sum_rate}")


if __name__ == '__main__':
    _main()



