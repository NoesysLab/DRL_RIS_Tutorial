import numpy as np
from scipy.constants import pi, speed_of_light
from dataclasses import dataclass, field
dcArray = lambda X: field(default=np.array(X))




################################# Configuration and Default Parameters #################################################

@dataclass()
class Setup:
    """
    Parameters of the Setup to be used.
    """

    K              : int        = 3                                               # Number of users
    S              : int        = 10                                              # Number of BS antennas
    M              : int        = 2                                               # Number of RIS
    N              : int        = 8                                               # Number of RIS elements
    N_tot          : int        = field(init=False)                               # Total number of RIS elements
    kappa_H        : float      = 5.                                              # BS-RIS Ricean factor (dB)
    kappa_G        : float      = 13.                                             # RIS-RX Ricean factor (dB)
    frequency      : float      = 32*10**9                                        # Operating frequency (Hz)
    wavelength     : float      = field(init=False)                               # wavelength (m)
    elem_dist      : float      = field(default='wavelength')                     # distance between elements in RIS
    RIS_resolution : int        = 2
    num_RIS_phases : int        = field(init=False)
    RIS_phases     : np.ndarray = dcArray([0, pi])
    precoding_v    : np.ndarray = field(default='ones')
    noise_variance : float      = 10.                                             # Variance of the AWGN - assumed equal for all RXs (dB)
    transmit_SNR   : float      = 1                                               # (dB)
    BS_position    : np.ndarray = dcArray([10, 0 , 2.])
    RIS_positions  : np.ndarray = dcArray([[5, 25, 2.], [15, 25, 2.]])
    RX_box         : np.ndarray = dcArray([[5., 20., 1.], [15., 30., 2.]])
    RX_positions   : np.ndarray = field(init=False)  
    d_BS_RIS       : np.ndarray = field(init=False)                               # Shape: (M,)
    d_RIS_RX       : np.ndarray = field(init=False)                               # Shape: (M, K)
    d_BS_RX        : np.ndarray = field(init=False)                               # Shape: (K)
    
    def __post_init__(self):
        """
        Initialization of Setup's parameters that depend on other user-defined values
        """
        self.N_tot          = self.N * self.M
        self.wavelength     = speed_of_light * self.frequency
        self.num_RIS_phases = 2 ** self.RIS_resolution
        self.elem_dist      = self.wavelength if self.elem_dist == 'wavelength' else self.elem_dist
        self.precoding_v    = np.ones(self.S) if self.precoding_v == 'ones' else self.precoding_v
        
        self.RX_positions   = get_random_RX_positions(self.K, self.RX_box[0, :], self.RX_box[1, :])
        self.d_BS_RIS       = calculate_distances(self.BS_position,   self.RIS_positions)  
        self.d_RIS_RX       = calculate_distances(self.RIS_positions, self.RX_positions)  
        self.d_BS_RX        = calculate_distances(self.BS_position,   self.RX_positions) 
        
        self.kappa_H        = dBW_to_Watt(self.kappa_H)
        self.kappa_G        = dBW_to_Watt(self.kappa_G)
        self.transmit_SNR   = dBW_to_Watt(self.transmit_SNR)
        self.noise_variance = dBm_to_Watt(self.noise_variance)




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

    return np.squeeze(dists)


def sample_gaussian_standard_normal(size=None):
    betta = np.random.normal(0, 1, size=size) + 1j * np.random.normal(0, 1, size=size)
    return betta





############################################ Channel Equations #########################################################


def calculate_pathloss(setup: Setup, dist):
    pl = dBW_to_Watt(-20 * np.log10( 4 * pi * dist / setup.wavelength ) )
    return pl

def calculate_array_response(setup: Setup, num_RIS_elements: int, RIS_position: np.ndarray, BS_or_RX_position: np.ndarray):
    """
    ULA model
    """
    dist      = calculate_distances(RIS_position, BS_or_RX_position)
    cos_angle = (RIS_position[0] - BS_or_RX_position[0]) / dist
    n         = np.arange(num_RIS_elements)
    a         = np.exp( (-1j * 2 * pi / setup.wavelength) * n * setup.elem_dist * cos_angle)

    assert a.shape[0] == setup.N
    return a



def calculate_H(setup: Setup, BS_position, RIS_position, dist_BS_RIS):
    pl             = calculate_pathloss(setup, dist_BS_RIS)
    LOS_component  = np.sqrt(setup.kappa_H/(setup.kappa_H+1)) * calculate_array_response(setup, setup.N, RIS_position, BS_position)
    LOS_component  = np.tile(LOS_component, (setup.S, 1))                         # Make array to (S,N) to account for multiple antennas
    NLOS_component = np.sqrt(1/(setup.kappa_H+1)) * sample_gaussian_standard_normal(size=(setup.S, setup.N))
    H              = np.sqrt(pl) * ( LOS_component + NLOS_component )

    assert H.shape[0] == setup.S and H.shape[1] == setup.N
    return H


def calculate_G(setup: Setup, RIS_position, RX_position, dist_RIS_RX):
    pl             = calculate_pathloss(setup, dist_RIS_RX)
    LOS_component  = np.sqrt(setup.kappa_G / (setup.kappa_G + 1)) * calculate_array_response(setup, setup.N, RIS_position, RX_position)
    NLOS_component = np.sqrt(1 / (setup.kappa_G + 1)) * sample_gaussian_standard_normal(size=setup.N)
    G              = np.sqrt(pl) * (LOS_component + NLOS_component)

    assert G.shape[0] == setup.N
    return G

def calculate_h(setup : Setup, dist_TX_RX):
    h =  np.sqrt(dist_TX_RX) * sample_gaussian_standard_normal(size=setup.S)
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


    signal_strenghts = np.zeros(setup.K, dtype=complex)
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



