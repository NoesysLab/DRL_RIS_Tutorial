from typing import Tuple

import numpy as np
from utils.custom_types import *



def normalize_array(a: np.array)->np.array:
    return (a - a.min())/(a.max()-a.min())


def reverse_normalize(a: np.array, min_: int, max_:int)->np.array:
    if max_ <= min_: raise ValueError
    if any(a>1) or any(a<0): raise ValueError
    return a*(max_-min_)+min_



def is_iterable(x)->bool:
    try:
        iter(x)
        return True
    except TypeError:
        return False

def convert2array(x: Union[int, float, np.ndarray, Iterable], shape: Union[int,Tuple[int,int]], dtype: np.dtype=None)->np.ndarray:
    """
    Depending on the value of 'x' return a corresponding numpy array
    :param x: The value(s) to be transformed to an array
    :param shape: Desired shape. If 'x' is already an array, it is reshaped
    :param dtype: Explicit dtype. If missing, numpy's decision is used.
    :return: If 'x' is a scalar number, return an array with 'x' in every element. If 'x' is an Iterable, cast in an array and reshape. If x is an array, reshape
    """
    if isinstance(x, np.ndarray):
        if x.shape != shape:
            x = x.reshape(shape)
            if dtype and x.dtype != dtype:
                x = x.astype(dtype)
        x_new = x

    elif is_iterable(x):
        x_new = np.array(x, dtype=dtype).reshape(shape)

    else:
        x_new = x*np.ones(shape, dtype)

    return x_new



def expand_array(a: np.ndarray, group_size: Tuple[int,int]):
    a_new = np.empty((a.shape[0]*group_size[0], a.shape[1]*group_size[1]), dtype=a.dtype)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            i_new_start = i*group_size[0]
            j_new_start = j*group_size[1]
            i_new_end   = i_new_start + group_size[0]
            j_new_end   = j_new_start + group_size[1]
            tile        = a[i,j]*np.ones(group_size, dtype=a.dtype)

            a_new[i_new_start:i_new_end, j_new_start:j_new_end] = tile

    return a_new


def condense_array(a: np.ndarray, group_size: Tuple[int, int]):

    if (a.shape % np.array(group_size)).sum() != 0:
        raise ValueError("dcArray shape not divisible by group sizes")

    row_indices = list(range(0, a.shape[0], group_size[0]))
    col_indices = list(range(0, a.shape[1], group_size[1]))

    i_list, j_list = np.meshgrid(row_indices, col_indices)
    return np.transpose(a[i_list, j_list])




def diag_per_row(M):
    b = np.zeros((M.shape[0], M.shape[1], M.shape[1]), dtype=M.dtype)
    diag = np.arange(M.shape[1])
    b[:, diag, diag] = M
    return b



def generate_binary_matrix(digits, start_from=None, end=None):
    B_initial = np.array([[0],[1]], dtype=np.byte)
    def next_column(B):
        k = B.shape[1]
        return np.block([[np.zeros((2**k,1), dtype=np.byte), B],[np.ones((2**k,1), dtype=np.byte),B]])
    B = B_initial
    for k in range(digits-1):
        B = next_column(B)



    return B



def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi
    return r, elev, az


def ray_to_elevation_azimuth(starting_point: Union[Vector3D, Matrix3DCoordinates],
                             ending_point: Union[Vector3D, Matrix3DCoordinates]) -> Union[Tuple[float,float], Tuple[Vector, Vector]]:

    v = ending_point - starting_point # type: np.ndarray
    if starting_point.ndim == 1:
        v = v.reshape((1,3))

    _, elev, az = cart2sph(v[:,0], v[:,1], v[:,2])

    if starting_point.ndim == 1:
        return float(elev), float(az)
    else:
        return elev, az



def safe_log10(A: np.ndarray)->np.ndarray:
    return np.log10(A, out=np.zeros_like(A), where=(A!=0))


def sample_gaussian_complex_matrix(shape: Tuple, mu=None, sigma=None):
    C =  np.random.randn(*shape) + 1j * np.random.randn(*shape)
    if sigma is not None:
        C *= sigma

    if mu is not None:
        C += mu

    return C

def dBm_to_Watt(val_dBm):
    return np.power(10, (val_dBm/10 - 3)  )

def dBW_to_Watt(val_dBW):
    return np.power(10, val_dBW/10)


def fmt_position(position: Vector3D)-> str:
    if position.dtype == int:
        return "({:d}, {:d}, {:d})".format(position[0],position[1],position[2])
    else:
        return "({:>.2f}, {:>.2f}, {:>.2f})".format(position[0], position[1], position[2])








################# K ###################
def lod_2_dol(l: List[Dict], numpy=False)->Union[Dict[str,List], Dict[str,np.ndarray]]:
    if len(l) == 0:
        return {}

    out_dict = dict()
    template_dict = l[0]
    for key in template_dict.keys():
        out_dict[key] = []

    for d in l:
        for key, value in d.items():
            out_dict[key].append(value)


    if numpy:
        for key in out_dict.keys():
            out_dict[key] = np.array(out_dict[key])

    return out_dict



def smooth_curve(y, window_size, order, deriv=0, rate=1):
    if window_size % 2 == 0: window_size += 1
    return savitzky_golay(y, window_size, order, deriv, rate)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.array([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b)[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

# def savitzky_golay_weights(window_size=None, order=2, derivative=0):
#     # The weights are in the first row
#     # The weights for the 1st derivatives are in the second, etc.
#     return savitzky_golay(window_size, order)[derivative]


