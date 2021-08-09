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
def exhaustive_search(H, G, h0, transmit_snr,
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