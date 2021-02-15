import numpy as np
from typing import *

# class ComplexNumber(np.core.multiarray):
#     pass
#
# class ExponentialComplex(ComplexNumber):
#     def __init__(self, magnitude: float, arg: float):
#         super().__init__(magnitude*np.exp(1j*arg))
#
# class PolarComplex(ComplexNumber):
#     def __init__(self, magnitude: float, arg: float):
#         super().__init__(magnitude*(np.cos(arg)+1j*np.sin(arg)))


def sample_gaussian_complex_matrix(shape: Tuple):
    return  np.random.randn(*shape) + 1j * np.random.randn(*shape)