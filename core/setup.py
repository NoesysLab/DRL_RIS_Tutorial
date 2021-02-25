from typing import *
import numpy as np
import yaml

from core.geometry import get_receiver_positions
from core.surfaces import RIS
from core.channels import RayleighFadeLink, Channel
from utils.custom_types import *

light_speed = 299792458  # Unit: m/s


class Setup:
    """
    A class that keeps all parameters of a system/setup in a single place.
    Useful for importing/exporting into files.
    """

    potential_numpy_properties = ['RIS_coordinates', 'TX_locations']
    optional_properties        = ['seed', 'test_RX_square_center', 'test_RX_square_width', 'test_RX_num_positions', 'test_RX_height']

    def __init__(self, setup_name=None, seed=42):
        # General parameters
        self.setup_name              = setup_name                                   # type: str
        self.seed                    = seed                                         # type: int


        # EM Waves (default values - change manually)
        self.carrier_frequency       = 32 * 10 ** 9  # Unit: Hz
        self.wavelength              = light_speed / float(self.carrier_frequency)  # type: float


        # Surfaces
        self.num_RIS                 = None                                         # type: int
        self.RIS_coordinates         = None                                         # type: Matrix3DCoordinates
        self.RIS_elements            = None                                         # type: Tuple[int, int]
        self.RIS_element_groups      = None                                         # type: Tuple[int, int]
        self.RIS_phase_values        = None                                         # type: Union[Iterable[float], Iterable[Complex]]


        # Surface Elements (default values - change manually)
        self.elements_correlation    = False
        self.element_dimensions      = (self.wavelength / 2, self.wavelength / 2)
        self.dependents_gap          = (self.wavelength, self.wavelength)
        self.independent_gap         = (self.wavelength, self.wavelength)


        # Transmitters
        self.TX_locations            = None                                         # type: Matrix3DCoordinates


        # Receivers (currently supporting one receiver placed on a number of position on a 2D square grid)
        self.train_RX_placement_type = 'grid'
        self.train_RX_square_center  = None                                         # type: Tuple[float, float]
        self.train_RX_square_width   = None                                         # type: float
        self.train_RX_num_positions  = None                                         # type: int
        self.train_RX_height         = None                                         # type: float

        self.test_RX_placement_type  = 'random'
        self.test_RX_square_center   = None                                         # type: Tuple[float, float]
        self.test_RX_square_width    = None                                         # type: float
        self.test_RX_num_positions   = None                                         # type: int
        self.test_RX_height          = None                                         # type: float


        # Channels
        self.noise_power             = None                                         # type: float
        self.TX_RIS_link_mult_factor = None                                         # type: float
        self.RX_RIS_link_mult_factor = None                                         # type: float
        self.TX_RX_link_mult_factor  = None                                         # type: float

        self.TX_RIS_link_is_LOS      = True
        self.RX_RIS_link_is_LOS      = True
        self.TX_RX_link_is_LOS       = True


        # Path Loss
        self.referenceDistance       = 1.0                                          # Unit: m
        self.pathlossCoefficientLOS  = 37.5                                         # Unit: dB
        self.pathlossExponentLOS     = 2.2                                          # Unit: dB
        self.pathlossCoefficientNLOS = 35.1                                         # Unit: dB
        self.pathlossExponentNLOS    = 3.67                                         # Unit: dB


    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get_config(self)->Dict[str, Dict]:
        """
        Get a nested dictionary containing the attributes of a Setup object.
        The outer dictionary has the setup name as a single key and the inner one
        has all the object attributes.
        Useful for exporting config files.
        """
        properties = vars(self)
        del properties['setup_name']

        for numpy_property in self.potential_numpy_properties:
            if isinstance(properties[numpy_property], np.ndarray):
                properties[numpy_property] = properties[numpy_property].to_list()

        return {self.setup_name: properties}


    def to_yaml(self, filename, mode='w'):
        """
        Save Setup to a YAML file.
        :param filename: The filename to write on.
        :param mode: To be passed in open()
        :return:
        """

        with open(filename, mode) as stream:
            try:
                data = self.get_config()
                config_dict: yaml.safe_dump(data, stream)

            except yaml.YAMLError as exc:
                print(exc)


    @staticmethod
    def from_config(config_dict: Dict[str, Dict]):
        """
        Create an instance of Setup from a dictionary. The dictionary must be nested.
        The outer dictionary must have the setup name as a single key and the inner one
        has all the object attributes.
        Useful for importing from config files.
        :param config_dict: A configuration dictionary with the properties of a Setup object.
        :return: A Setup object.
        """
        setup_name = list(config_dict.keys())[0]
        properties = list(config_dict.values())[0] # type: Dict
        obj        = Setup(setup_name)

        for attr_name, attr_value in properties.items():
            obj[attr_name] = attr_value

        for numpy_property in obj.potential_numpy_properties:
            obj[numpy_property] = np.array(obj[numpy_property])

        required_properties = set(obj.__dict__.keys()) - set(obj.optional_properties)
        for prop in required_properties:
            if obj[prop] is None:
                raise ValueError("Required property '{}' is not set in Setup config file.".format(prop))



        return obj


    @staticmethod
    def from_yaml_file(filename: str):
        """
        Create a Setup instance from a YAML configuration file.
        :param filename: The file with an appropriate format (same as the dict returned from Setup.get_config() ). The filename extension should be included.
        :return: A Setup object
        """


        with open(filename, 'r') as stream:
            try:
                config_dict: yaml.safe_load(stream)
                return Setup.from_config(config_dict)

            except yaml.YAMLError as exc:
                print(exc)




def initialize_simulation_from_setup(setup: Setup)->Tuple[List[RIS], Tuple[Channel, Dict], Tuple[Channel, Dict], Tuple[Channel, Dict], Matrix3DCoordinates, Matrix3DCoordinates]:
    """
    Construct appropriate objects from a configurations to start the simulation.
    This function encapsulates all of the initialization code, apart from the channels which change with coordinates.
    :param setup: A Setup objects with the current parameters
    :return: A Tuple:
               i)   A list of RIS,
               ii)  Class and parameters for the TX_RIS link to be passed to a Channel object,
               iii) Class and parameters for the RX_RIS link to be passed to a Channel object,
               iv)  Class and parameters for the TX_RX link to be passed to a Channel object,
               v)   Positions for the RX for training phase,
               vi)  Positions for the RX for testing phase
    """


    RIS_list = []
    for i in range(setup.num_RIS):
        ris = RIS(position              = setup.RIS_coordinates[0, :],
                  element_grid_shape    = setup.RIS_elements,
                  element_group_size    = setup.RIS_element_groups,
                  element_dimensions    = setup.element_dimensions,
                  in_group_spacing      = setup.dependents_gap,
                  between_group_spacing = setup.independent_gap,
                  phase_space           = ('discrete', {'values': setup.RIS_phase_values}),
                  id_                   = i)

        ris.set_random_state()
        RIS_list.append(ris)


    TX_RIS_link_info = (RayleighFadeLink, {'mult_factor': setup.TX_RIS_link_mult_factor,
                                           'isLOS': setup.TX_RIS_link_is_LOS})
    RX_RIS_link_info = (RayleighFadeLink, {'mult_factor': setup.RX_RIS_link_mult_factor,
                                           'isLOS': setup.RX_RIS_link_is_LOS})
    TX_RX_link_info  = (RayleighFadeLink, {'mult_factor': setup.TX_RX_link_mult_factor,
                                           'isLOS': setup.TX_RX_link_is_LOS})

    train_RX_locations = get_receiver_positions(setup.train_RX_placement_type,
                                                 setup.train_RX_num_positions,
                                                 setup.train_RX_square_center,
                                                 setup.train_RX_square_width,
                                                 setup.train_RX_height)
    if setup.test_RX_square_center is not None:
        test_RX_locations = get_receiver_positions(setup.test_RX_placement_type,
                                                     setup.test_RX_num_positions,
                                                     setup.test_RX_square_center,
                                                     setup.test_RX_square_width,
                                                     setup.test_RX_height)
    else:
        test_RX_locations = None

    return RIS_list, TX_RIS_link_info, RX_RIS_link_info, TX_RX_link_info, train_RX_locations, test_RX_locations


























