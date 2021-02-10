import pytest
import numpy as np
import warnings


from utils.misc import *

def test_normalize1():
    a = np.logspace(1,10)
    a_norm = normalize_array(a)

    assert a_norm.max() == 1
    assert a_norm.min() == 0


def test_normalize_nan():
    a = np.logspace(1, 100000) # 10**10000 is Inf
    a_norm = normalize_array(a)

    assert np.isnan(a_norm.max())
    assert np.isnan(a_norm.min())

def test_normalize_all_equal_array():
    np.seterr(all='raise')
    with pytest.raises(FloatingPointError) as e:
        a = 42*np.ones(100)
        a_norm = normalize_array(a)
        assert 'invalid value encountered in true_divide' in str(e)


def test_reverse_normalize():
    a = np.linspace(12, 21, num=57)
    a_new = reverse_normalize(normalize_array(a), a.min(), a.max())
    assert np.array_equal(a_new, a)


def test_reverse_normalize_error_min_max():
    with pytest.raises(ValueError):
        a = []
        a_new = reverse_normalize([], 2, 1)

def test_reverse_normalize_unnormalized_values():
    with pytest.raises(ValueError):
        a = np.linspace(12, 21, num=57)
        a_new = reverse_normalize(a, a.min(), a.max())

    with pytest.raises(ValueError):
        a = np.linspace(-100, 21, num=57)
        a_new = reverse_normalize(a, a.min(), a.max())





def test_expand_array():
    A = np.array([
        [ 0, 3],
        [12, 15],
        [24, 27],
        [36, 39],])

    group_size = (2,3)
    A_result = np.array([
        [ 0,  0,  0,  3,  3,  3],
        [ 0,  0,  0,  3,  3,  3],
        [12, 12, 12, 15, 15, 15],
        [12, 12, 12, 15, 15, 15],
        [24, 24, 24, 27, 27, 27],
        [24, 24, 24, 27, 27, 27],
        [36, 36, 36, 39, 39, 39],
        [36, 36, 36, 39, 39, 39],
    ])


    A_new = expand_array(A, group_size)

    assert np.array_equal(A_new, A_result)


def test_condense_array():
    A = np.arange(0, 48).reshape(8, 6)
    group_size = (2,3)

    A_new = condense_array(A, group_size)

    assert np.array_equal(A_new, np.array([[ 0, 3],
                                           [12, 15],
                                           [24, 27],
                                           [36, 39],]))
