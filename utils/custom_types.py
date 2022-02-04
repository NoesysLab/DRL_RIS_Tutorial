import numpy as np
from typing import *


# type aliases for more descriptive type hints in function arguments


Vector              = NewType('Vector'  , np.ndarray)              # An 1D array
Matrix              = NewType('Matrix'  , np.ndarray)              # A  2D array
Vector2D            = NewType('Vector2D', np.ndarray)              # An 1D array of length 2
Vector3D            = NewType('Vector3D', np.ndarray)              # An 1D array of length 3
Matrix2D            = NewType('Matrix2D', np.ndarray)              # A  2D array of shape (N, 2)
Matrix3D            = NewType('Matrix3D', np.ndarray)              # A  3D array of shape (N, 3)
Matrix3DCoordinates = NewType('Matrix3DCoordinates', np.ndarray)   # A  3D array of shape (N, 3) where each row may be interpreted as 3D coordinates
Complex             = NewType('Complex', Tuple[float,float])       # A complex number as a tuple of real and imaginary parts
ComplexVector       = NewType('ComplexVector', np.ndarray)         # An 1D array of dtype=complex
ComplexMatrix2D     = NewType('ComplexMatrix2D', np.ndarray)       # A  2D array of dtype=complex
ComplexMatrix3D     = NewType('ComplexMatrix3D', np.ndarray)       # A  3D array of dtype=complex
ComplexArray        = NewType('ComplexArray'   , np.ndarray)       # A numpy array of dtype=complex and arbitrary shape


VarsString          = NewType('VarsString', str)