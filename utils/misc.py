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


