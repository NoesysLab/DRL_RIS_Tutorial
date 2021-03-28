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
        raise ValueError("Array shape not divisible by group sizes")

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
