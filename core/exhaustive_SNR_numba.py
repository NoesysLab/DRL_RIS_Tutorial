from typing import Tuple

import numba
import numpy as np


@numba.njit
def nb_block(X):
    xtmp1 = np.concatenate(X[0], axis=1)
    xtmp2 = np.concatenate(X[1], axis=1)
    return np.concatenate((xtmp1, xtmp2), axis=0)


@numba.njit
def nb_block2(X):
    xtmp1 = np.hstack(X[0])
    xtmp2 = np.hstack(X[1])
    return np.vstack((xtmp1, xtmp2))


@numba.jitclass([
    ('B', numba.int8[:,:]),
    ('k', numba.int64),
    ('num_digits', numba.int64),
    ('max_number', numba.float64),
    ('curr_padding', numba.int64),
    ('_curr_num', numba.int64),
])
class BinaryEnumerator:


    def __init__(self, num_digits):
        self.B            = np.array([[0],[1]], dtype=np.byte)
        self.k            = 1
        self.num_digits   = num_digits
        self.max_number   = np.power(2, num_digits)-1
        self.curr_padding = self.num_digits - self.k

        self._curr_num   = 0


    def _expand_array(self):

        self.B = nb_block2(((np.zeros((2 ** self.k, 1), dtype=np.byte), self.B),
                           (np.ones((2 ** self.k, 1), dtype=np.byte), self.B)))

        self.k += 1
        self.curr_padding = self.num_digits - self.k


    # def __iter__(self):
    #     self._curr_num = 0
    #     return self


    def next(self):
        if self._curr_num > self.max_number:
            raise StopIteration


        if self._curr_num + 1 > np.power(2, self.k):
            self._expand_array()


        num = self.B[self._curr_num, :]
        self._curr_num += 1


        if self.curr_padding>0:
            num = np.concatenate( ( np.zeros(self.curr_padding, dtype=np.byte), num ) )

        return num





@numba.vectorize(nopython=True)
def to_complex(x):
    return x+0j






@numba.njit(fastmath=False)
def exhaustive_SNR_search(H, G, h0, transmit_snr,
                          total_tunable_elements,
                          total_dependent_elements,
                          phase_values, )->Tuple[np.ndarray, float]:

    num_discrete_states     = len(phase_values)
    configurations_iterator = BinaryEnumerator(total_tunable_elements)
    num_configurations      = int(2**total_tunable_elements)

    best_snr                = 0.
    best_configuration      = np.zeros(num_discrete_states, dtype=np.byte)


    assert num_discrete_states == 2, 'Currently supporting only two discrete states due to custom enumeration of the configuration/phase space.'


    for i in range(num_configurations):

        configuration                  = configurations_iterator.next().flatten()
        phase                          = phase_values[configuration]
        Phi                            = np.repeat(phase, repeats=total_dependent_elements)
        Phi2                           = to_complex(Phi).flatten()
        prod                           = (G.T*H).flatten()
        channel_reflected              = np.dot(prod, Phi2) + h0
        channel_reflected2             = channel_reflected[0,0]
        channel_mag                    = np.power( np.absolute(channel_reflected2), 2)
        snr                            = channel_mag * transmit_snr

        if snr > best_snr:
            best_snr           = snr
            best_configuration = configuration

    return best_configuration, best_snr


@numba.njit(fastmath=False)
def exhaustive_SINR_search(H : np.ndarray,
                           G : np.ndarray,
                           h0 : np.ndarray,
                           codebook : np.ndarray,
                           total_tunable_elements : int,
                           total_dependent_elements : int,
                           phase_values: np.ndarray) -> Tuple[int, np.ndarray, float]:
    """
    Let
        `B` : number of BS antennas
        `K` : number of users
        `M` : number of RIS
        `N` : number of RIS elements (of a single RIS)
        'V' : The size of the codebook (number of precoding matrices)

    :param H: (M, B, N)
    :param G: (M, K, N)
    :param h0: (K, B)
    :param codebook: (V, K, B)
    :param total_tunable_elements:
    :param total_dependent_elements:
    :param phase_values:
    :return: (best_precoding_matrix_selection, best_RIS_profile, best_SINR_value)
    """

    M, B, N = H.shape
    K       = G.shape[1]
    V       = codebook.shape[0]

    num_discrete_states     = len(phase_values)
    configurations_iterator = BinaryEnumerator(total_tunable_elements)
    num_configurations      = int(2**total_tunable_elements)

    best_sinr                     = 0.
    best_configuration            = np.zeros(num_discrete_states, dtype=np.byte)
    best_precoding_matrix_index   = -1


    assert num_discrete_states == 2, 'Currently supporting only two discrete states due to custom enumeration of the configuration/phase space.'


    for i in range(num_configurations):

        signal_strenghts = np.zeros(K, dtype=np.float32)
        Phi_             = np.zeros((M, N, N))
        Phi              = to_complex(Phi_)

        for m in range(M):
            Phi[m, :, :] = np.diag(RIS_profiles[m, :])


        configuration                  = configurations_iterator.next().flatten()
        phase                          = phase_values[configuration]
        Phi                            = np.repeat(phase, repeats=total_dependent_elements)
        Phi2                           = to_complex(Phi).flatten()
        prod                           = (G.T*H).flatten()
        channel_reflected              = np.dot(prod, Phi2) + h0
        channel_reflected2             = channel_reflected[0,0]
        channel_mag                    = np.power( np.absolute(channel_reflected2), 2)
        snr                            = channel_mag * transmit_snr

        if snr > best_snr:
            best_snr           = snr
            best_configuration = configuration

    return best_configuration, best_snr