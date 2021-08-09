import numpy as np
from scipy.constants import pi, speed_of_light

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


def calculate_pathloss(dist):
    global wavelength
    pl = dBW_to_Watt(-20 * np.log10( 4 * pi * dist / wavelength ) )
    return pl

def calculate_array_response(num_RIS_elements: int,
                             RIS_position: np.ndarray,
                             BS_or_RX_position: np.ndarray):
    '''
    ULA model
    '''
    global elem_dist, wavelength
    dist      = calculate_distances(RIS_position, BS_or_RX_position)
    cos_angle = (RIS_position[0] - BS_or_RX_position[0]) / dist
    n         = np.arange(num_RIS_elements)
    a         = np.exp( (-1j * 2 * pi / wavelength) * n * elem_dist * cos_angle)
    return a

def sample_gaussian_standard_normal(size=None):
    betta = np.random.normal(0, 1, size=size) + 1j * np.random.normal(0, 1, size=size)
    return betta

def calculate_H(BS_position, RIS_position, dist_BS_RIS):
    global S, N, kappa_H
    pl             = calculate_pathloss(dist_BS_RIS)
    LOS_component  = np.sqrt(kappa_H/(kappa_H+1)) * calculate_array_response(N, RIS_position, BS_position)
    LOS_component  = np.tile(LOS_component, (S, 1))                         # Make array to (S,N) to account for multiple antennas
    NLOS_component = np.sqrt(1/(kappa_H+1)) * sample_gaussian_standard_normal(size=(S,N))
    H              = np.sqrt(pl) * ( LOS_component + NLOS_component )

    assert H.shape[0] == S and H.shape[1] == N
    return H

def calculate_G(RIS_position, RX_position, dist_RIS_RX):
    global N, kappa_G
    pl             = calculate_pathloss(dist_RIS_RX)
    LOS_component  = np.sqrt(kappa_G / (kappa_G + 1)) * calculate_array_response(N, RIS_position, RX_position)
    NLOS_component = np.sqrt(1 / (kappa_G + 1)) * sample_gaussian_standard_normal(size=N)
    G              = np.sqrt(pl) * (LOS_component + NLOS_component)

    assert G.shape[0] == N
    return G

def calculate_h(dist_TX_RX):
    global S
    h =  np.sqrt(dist_TX_RX) * sample_gaussian_standard_normal(size=S)
    assert h.shape[0] == S
    return h


def simulate_transmission():
    global K, S, M, N, BS_position, RIS_positions, RX_positions, d_BS_RIS, d_BS_RX, d_RIS_RX

    H = np.empty(shape=(M, S, N), dtype=complex)
    G = np.empty(shape=(M, K, N), dtype=complex)
    h = np.empty(shape=(K, S),    dtype=complex)


    for m in range(M):
        H[m,:,:] = calculate_H(BS_position, RIS_positions[m,:], d_BS_RIS[m])

    for k in range(K):
        h[k, :] = calculate_h(d_BS_RX[k])

        for m in range(M):
            G[m,k,:] = calculate_G(RIS_positions[m,:], RX_positions[k,:], d_RIS_RX[m,k])

    return H, G, h


def compute_SINR_per_user(noise_variance,
                         H,
                         G,
                         h,
                         RIS_profiles,
                         precoding_vector):
    '''
    Compute the SNR observed at every receiver

    :param transmit_SNR: In Watt
    :param noise_variance: In watt
    :param H: (M, S, N)
    :param G: (M, K, N)
    :param h: (K, S)
    :param RIS_profiles: (M, N)
    :param precoding_vector: (N,)
    :return: (K,)
    '''

    global K, S, M, N

    signal_strenghts = np.zeros(K, dtype=complex)
    Phi              = np.zeros((M, N, N), dtype=complex)

    for m in range(M):
        Phi[m, :, :] = np.diag(RIS_profiles[m,:])

    for k in range(K):
        total_RIS_cascaded_channel = np.zeros(S, dtype=complex)
        for m in range(M):
            H_m   = H[m,:,:]    # (S, N)
            Phi_m = Phi[m,:,:]  # (N, N)
            G_m   = G[m,k,:]    # (N, )
            G_m   = G_m.reshape((N, 1))
            this_RIS_cascaded_channel =  H_m @ Phi_m @ G_m
            total_RIS_cascaded_channel += this_RIS_cascaded_channel.flatten()


        signal_strength = np.dot(h[k] + total_RIS_cascaded_channel, precoding_vector)
        signal_strength = np.absolute(signal_strength) ** 2

        signal_strenghts[k] = signal_strength

    sum_signal_strengths = np.sum(signal_strenghts)
    SINR = np.zeros(K)
    for k in range(K):
        SINR[k] = signal_strenghts[k] / (noise_variance + sum_signal_strengths - signal_strenghts[k])

    return SINR


def compute_sum_rate(SINR):
    return np.sum( np.log2(1 + SINR) )


def RIS_state2profile(theta):
    return np.exp(-1j * pi * theta)



K              = 3                             # Number of users
S              = 10                            # Number of BS antennas
M              = 2                             # Number of RIS
N              = 8                             # Number of RIS elements
N_tot          = N * M                         # Total number of RIS elements
kappa_H        = 5                             # BS-RIS Ricean factor (dB)
kappa_G        = 13                            # RIS-RX Ricean factor (dB)
frequency      = 32*10**9                      # Operating frequency (Hz)
wavelength     = speed_of_light / frequency    # wavelength (m)
elem_dist      = wavelength                    # distance between elements in RIS
RIS_resolution = 2
num_RIS_phases = 2**RIS_resolution
RIS_phases     = np.array([0, pi])
precoding_v    = np.array([1]*S)
noise_variance = 10                           # Variance of the AWGN - assumed equal for all RXs (dB)
transmit_SNR   = 1                              # (dB)
BS_position    = np.array( [10,  0 , 2.])
RIS_positions  = np.array([[5,   25, 2.],
                           [15,  25, 2.]])
RX_box         = np.array([[5.,  20., 1.],
                           [15., 30., 2.]])
RX_positions   = get_random_RX_positions(K, RX_box[0,:], RX_box[1,:])
d_BS_RIS       = calculate_distances(BS_position, RIS_positions)       # Shape: (M,)
d_RIS_RX       = calculate_distances(RIS_positions, RX_positions)      # Shape: (M, K)
d_BS_RX        = calculate_distances(BS_position, RX_positions)        # Shape: (K)


kappa_H        = dBW_to_Watt(kappa_H)
kappa_G        = dBW_to_Watt(kappa_G)
transmit_SNR   = dBW_to_Watt(transmit_SNR)
noise_variance = dBm_to_Watt(noise_variance)



thetas       = np.random.choice(RIS_phases, size=(M, N))
RIS_profiles = RIS_state2profile(thetas)
H, G, h      = simulate_transmission()
SINR         = compute_SINR_per_user(noise_variance, H, G, h, RIS_profiles, precoding_v)
sum_rate     = compute_sum_rate(SINR)

print(f"Sum rate: {sum_rate}")



