import itertools

import numpy as np
from typing import *
from utils.misc import convert2array, expand_array, Vector3D, Vector2D, Matrix3DCoordinates
import gym.spaces as spaces
import functools








############################ StateSpaces ##########################################
"""
A hierarchy of allowed spaces for the values of the internal state of an RIS.
Those are simply wrappers on the gym.spaces classes. 
"""

'''Base State Space type. Only useful for type hints'''
StateSpace = spaces.Space



class DiscreteStateSpace(spaces.MultiDiscrete):
    '''A multi-dimensional space of discrete values {0,1,...} (same for each dimension)'''
    def __init__(self, dimension: int, num_values: int, *args, **kwargs):
        self.dimension = dimension
        self.num_values = num_values
        super().__init__([num_values]*dimension)

class BinaryStateSpace(spaces.MultiBinary):
    '''A multi-dimensional binary space'''
    def __init__(self, dimension: int, *args, **kwargs):
        self.dimension = dimension
        self.num_values = 2
        super().__init__(dimension)

class ContinuousStateSpace(spaces.Box):
    '''A continuous space with potentially different bounds in each dimension'''
    def __init__(self, low: List[float], high: List[float], shape=None, dtype=np.float32, *args, **kwargs):
        super().__init__(low, high, shape, dtype)




def StateSpaceFactory(name:str, *args, **kwargs)->StateSpace:
    """
    This is a factory method for constructing a StateSpace from a given name.
    :param name: str. One of {'binary', 'discrete', 'continuous'}
    :param args: Positional arguments to be passed to the appropriate constructor
    :param kwargs: Keyword arguments to be passed to the appropriate constructor
    :return: The corresponding StateSpace subtype object
    """
    spaces = {
        'binary'    : BinaryStateSpace,
        'discrete'  : DiscreteStateSpace,
        'continuous': ContinuousStateSpace,
    }
    try:
        return spaces[name.lower()](*args, **kwargs)
    except KeyError:
        raise ValueError("Unrecognized state space name '{}'".format(name))





############################ PhaseSpaces ############################################
"""
A hierarchy of classes that are used to convert the internal RIS states to effective phase shifts.
"""

class PhaseSpace:
    '''Base method of all Phase spaces'''

    def calculate_phase_shifts(self, state: np.array)->np.ndarray:
        '''Convert a StateSpace array to the appropriate phase shifts values. To be implemented by the subclasses'''
        raise NotImplementedError



class DiscretePhaseSpace(PhaseSpace):
    """
    A discrete phase states where each element can take one of a predefined set of values
    """
    def __init__(self, values: List):
        """
        Construct a DiscretePhaseStates object.
        :param values: The discrete phase shift values.
        """
        self.values = values

    def calculate_phase_shifts(self, state: np.array)->np.ndarray:
        """
        '''Convert a StateSpace array to the appropriate phase shifts values.'''
        :param state: The array of (discrete) RIS states. They are assumed to have a 1-1 correspondence to the phase shift values. It can be of arbitrary shape.
        :return: An array of the same shape with the states mapped to discrete values
        """
        def state2phase(s):
            return self.values[s]
        map_states_to_phase_shifts = np.vectorize(state2phase)
        return map_states_to_phase_shifts(state)


def PhaseSpaceFactory(name:str, *args, **kwargs):
    """
        This is a factory method for constructing a PhaseSpace from a given name.
        :param name: str. One of {'discrete'}
        :param args: Positional arguments to be passed to the appropriate constructor
        :param kwargs: Keyword arguments to be passed to the appropriate constructor
        :return: The corresponding StateSpace subtype object
        """
    spaces = {
        'discrete'  : DiscretePhaseSpace,
    }
    try:
        return spaces[name.lower()](*args, **kwargs)
    except KeyError:
        raise ValueError("Unrecognized phase space name '{}'".format(name))















##################### Surfaces ############################################################



class RIS:
    """
    A class that implements the Reconfigurable Intelligent Surface functionality.
    It holds its own 3D position, its elements positions, its internal state and the corresponding phase shifts.
    It supports a 2D grid of elements (aligned on the Z plane), which can be grouped into 2D sub-grids of dependent (same-state) elements.
    """
    def __init__(self,
                 position             : Vector3D,
                 element_grid_shape   : Tuple[int, int],
                 element_group_size   : Union[None, Tuple[int, int]],
                 element_dimensions   : Union[list, Vector2D, int],
                 in_group_spacing     : Union[list, Vector2D, int],
                 between_group_spacing: Union[list, Vector2D, int],
                 phase_space          : Tuple[str, Dict],
                 id_=None):
        """
        Construct an RIS object.
        :param position: A 3D vector of x,y,z coordinates of its position.
        :param element_grid_shape: A tuple of (number_of_rows, number_of_columns) for the internal element grid
        :param element_group_size: A tuple of (number_of_rows, number_of_columns) for each subgroup of dependent elements.
        :param element_dimensions: A tuple of (width, height) with the dimensions of each element.
        :param in_group_spacing:   A tuple of (width, height) specifying the spacing between consecutive elements in the same group.
        :param between_group_spacing: A tuple of (width, height) specifying the spacing between the ending and starting elements between consecutive groups.
        :param phase_space: A PhaseSpace object that signifies allowed values for the effective time shifts.
        :param id_: An identification token for this specific RIS. If not provided, python's id(self) will be used.
        """

        if not hasattr(position, '__len__') or len(position) != 3:
            raise ValueError("Expected a 3D position vector")

        self.shape                = element_grid_shape                                              # (rows,cols) for the grid of elements
        self.group_shape          = element_group_size if element_grid_shape else (1,1)             # (rows,cols) for each subgroup of the grid of elements
        self.element_dimensions   = element_dimensions                                              # (width,height) for each element
        self.position             = position                                                        # 3D position of the RIS (equals to the position of element (0,0) )
        self.id                   = id_ if id_ is not None else id(self)                            # Useful for printing/plotting
        self.total_elements       = np.prod(element_grid_shape)                                     # Number of elements within the grid (some may be dependent - i.e. always have the same state)
        self.num_tunable_elements = self.total_elements // np.prod(element_group_size)              # Number of elements whose state can be set individually. This is the number of groups in the grid.
        self.phase_space          = PhaseSpaceFactory(phase_space[0], **phase_space[1])             # Used to map internal states to phase shifts
        self.state_space          = StateSpaceFactory(phase_space[0], dimension=self.num_tunable_elements, num_values=len(self.phase_space.values), **phase_space[1])  # Allowed values for internal state. Used only for checking when setting values and initializing to random state
        self.element_coordinates  = self.construct_element_coordinates_array(self.position,         # A 2D matrix of shape (total_elements, 3) with the 3D position of each element (Z values are the same)
                                                                             element_grid_shape,
                                                                             element_dimensions,
                                                                             element_group_size,
                                                                             in_group_spacing,
                                                                             between_group_spacing)
        # The internal state of the RIS, kept as a 1D array.
        # **IMPORTANT: This variable only keeps the state values for the tunable elements.
        # This convention should be kept when setting new values and when reading its value.**
        self._state               = None                                                            # type: np.ndarray # It allways belongs to self.state_space






    @staticmethod
    def construct_element_coordinates_array(base_coordinates     : Vector3D,
                                            shape                : Tuple[int, int],
                                            element_dimensions   : Union[list, Vector2D, int],
                                            group_size           : Union[None, Tuple[int, int]],
                                            in_group_spacing     : Union[list, Vector2D, int],
                                            between_group_spacing: Union[list, Vector2D, int]) -> Matrix3DCoordinates:
        '''For each element in the grid, calculate its coordinates based on the parameters passed to the constructor.
        Only used once by the __init__ method.'''

        if shape[0] % group_size[0] != 0 or shape[1] % group_size[1] != 0:
            raise ValueError("Element grid shape is not divisible by group size.")

        in_group_spcaing      = convert2array(in_group_spacing, 2)
        between_group_spacing = convert2array(between_group_spacing, 2)
        element_dimensions    = convert2array(element_dimensions, 2)

        def _calculate_element_coordinates(i: int, j: int) -> Vector3D:
            num_elements_before = np.array([i, j])
            num_groups_before = num_elements_before // group_size
            num_in_group_spacings_needed = num_elements_before - num_groups_before

            coordinates = base_coordinates[0:2] + \
                          num_elements_before * element_dimensions + \
                          num_groups_before * between_group_spacing + \
                          num_in_group_spacings_needed * in_group_spcaing

            coordinates3D = np.hstack([coordinates, base_coordinates[2]])
            return Vector3D(coordinates3D)


        coords_table = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                coords_table.append(_calculate_element_coordinates(i, j))

        return np.array(coords_table)


    def set_random_state(self):
        """
        Set the internal state as a random sample from the StateSpace of the RIS.
        Useful for initialization.
        :return: A pointer to this object for chaining commands
        """
        self._state = self.state_space.sample()
        return self

    def get_state(self)->np.ndarray:
        """
        Get the internal state of the RIS. Note that this is a matrix that accounts only for the
        tunable elements, rather than the total number of elements within the grid.
        :return: An array representation of the RIS state. It can be or arbitrary shape but normally it is 1D.
        """
        if self._state is None: raise RuntimeError("Uninitialized RIS state.")
        return self._state

    def set_state(self, new_state: Union[np.array, list]):
        """
        Change the internal state of the RIS.
        Note that passed array is assumed to have a size equal to the number of TUNABLE elements in the grid,
        rather than the total elements. Dependent elements are handled internally.
        :param new_state: The desired state as an array. It must belong to the specified StateSpace of the object.
        """
        if new_state not in self.state_space:
            raise ValueError("Setting state to {} is invalid for a state space {}".format(new_state, type(self.state_space).__class__))

        self._state = new_state


    def get_phase(self, form='2D'):
        """
        Get the phase shifts induced by this RIS. Those are computed through the internal state and the PhaseSpace
        specified for this object. The resulting array contains phase shifts both for the tunable and depended elements.
        :param form: str. One of {'1D', '2D'}. The dimensionality of the returned phase array. Repeated values for depended elements re handled differently in each case.
        :return: An array of the calculated phase shifts.
        """
        num_dependent_elements_per_group = np.prod(self.group_shape)
        phase_shifts = self.phase_space.calculate_phase_shifts(self.get_state())

        if form == '1D':
            return np.repeat(phase_shifts, num_dependent_elements_per_group)

        elif form == '2D':
            matrix_shape = (self.shape[0]//self.group_shape[0], self.shape[1]//self.group_shape[1])
            matrix       = phase_shifts.reshape(matrix_shape)
            matrix       = expand_array(matrix, self.group_shape)
            return matrix
        else:
            raise ValueError("Expected one of {{'1D', '2D'}} for parameter 'form' value. Got {} instead.".format(form))


    def get_element_coordinates(self)->Matrix3DCoordinates:
        """
        Get all coordinates of the elements of the grid of the RIS.
        :return: Array of shape (num_elements, 3) where each row is the x,y,z coordinates for each element
        """
        return self.element_coordinates



    # def get_element_coordinates_box(self):
    #     """
    #     Get a box that includes
    #     :return:
    #     """
    #     return np.array([
    #         [self.element_coordinates[:, :, 0].min(), self.element_coordinates[:, :, 0].max()+self.element_coordinates[0]],
    #         [self.element_coordinates[:, :, 1].min(), self.element_coordinates[:, :, 1].max()+self.element_coordinates[1]],
    #         [self.element_coordinates[:, :, 2].min(), self.element_coordinates[:, :, 2].max()],
    #     ])





