import numpy as np
from typing import *
# type aliases for more descriptive type hints in function arguments
Vector              = NewType('Vector'  , np.ndarray)
Matrix              = NewType('Matrix'  , np.ndarray)
Vector2D            = NewType('Vector2D', np.ndarray)
Vector3D            = NewType('Vector3D', np.ndarray)
Matrix2D            = NewType('Matrix2D', np.ndarray)
Matrix3DCoordinates = NewType('Matrix3DCoordinates', np.ndarray)
Complex             = NewType('Complex', Tuple[float,float])