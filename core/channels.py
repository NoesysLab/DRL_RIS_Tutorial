

from numpy import sin, cos, pi, sqrt, sum, floor, ceil, power, log10, exp
import numpy as np
from typing import *
from itertools import product
import warnings





from core.surfaces import RIS
from utils.custom_configparser import CustomConfigParser
from utils.custom_types import Vector, Matrix2D, Matrix3D, ComplexVector, ComplexArray, Vector3D, Matrix3DCoordinates
from utils.misc import safe_log10, sample_gaussian_complex_matrix, dBW_to_Watt, ray_to_elevation_azimuth

wall_attenuation = None

lightspeed = None
frequency  = None
wavelength = None

k          = None
q          = None

f0_LOS     = None
n_LOS      = None
b_LOS      = None
sigma_LOS  = None

f0_NLOS    = None
n_NLOS     = None
b_NLOS     = None
sigma_NLOS = None




l_h                       = None
l_g                       = None
l_SISO                    = None
shadow_fading_exists      = None
normalize_steering_vector = None
normalize_Ge              = None
ignore_LOS                = None
TX_RX_mult_factor         = None
element_spacing           = None
pathloss_db_sign          = None
units_scale               = None

rng = None


def initialize_from_config(config: CustomConfigParser):
    global rng
    seed = config.getint('program_options', 'random_seed')
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()


    global f0_LOS, n_LOS, b_LOS, sigma_LOS, f0_NLOS, n_NLOS, b_NLOS, sigma_NLOS, lightspeed, frequency, q, wavelength, k, wall_attenuation
    global l_h, l_g, l_SISO, shadow_fading_exists,normalize_steering_vector, normalize_Ge, ignore_LOS, TX_RX_mult_factor
    global element_spacing, pathloss_db_sign, units_scale


    if config.get('setup', 'environment_type') == 'indoor':
        section = 'pathloss_indoor'
    elif config.get('setup', 'environment_type') == 'outdoor':
        section = 'pathloss_outdoor'
    else:
        raise ValueError

    f0_LOS     = config.getfloat(section, 'f0_LOS')
    n_LOS      = config.getfloat(section, 'n_LOS')
    b_LOS      = config.getfloat(section, 'b_LOS')
    sigma_LOS  = config.getfloat(section, 'sigma_LOS')
    f0_NLOS    = config.getfloat(section, 'f0_NLOS')
    n_NLOS     = config.getfloat(section, 'n_NLOS')
    b_NLOS     = config.getfloat(section, 'b_NLOS')
    sigma_NLOS = config.getfloat(section, 'sigma_NLOS')

    lightspeed = config.getfloat('constants', 'speed_of_light')
    frequency  = config.getfloat('setup', 'frequency')
    q          = config.getfloat('channel_modeling', 'q')

    wavelength = lightspeed / frequency
    k          = 2 * pi / wavelength

    wall_attenuation          = config.getfloat('channel_modeling', 'wall_attenuation')
    l_h                       = config.getfloat('channel_modeling', 'l_h')
    l_g                       = config.getfloat('channel_modeling', 'l_g')
    l_SISO                    = config.getfloat('channel_modeling', 'l_SISO')
    shadow_fading_exists      = config.getboolean('channel_modeling', 'shadow_fading_exists')
    normalize_steering_vector = config.getboolean('channel_modeling', 'normalize_steering_vector')
    normalize_Ge              = config.getboolean('channel_modeling', 'normalize_Ge')
    ignore_LOS                = config.getboolean('channel_modeling','ignore_LOS')
    TX_RX_mult_factor         = config.getfloat('channel_modeling', 'TX_RX_mult_factor')
    pathloss_db_sign          = config.get('channel_modeling', 'pathloss_db_sign')
    units_scale                =  config.get('channel_modeling', 'units_scale')

    if units_scale == 'power':
        l_h = dBW_to_Watt(l_h)
        l_g = dBW_to_Watt(l_g)
        if l_SISO != 0:
            l_SISO = dBW_to_Watt(l_SISO)


    element_spacing = wavelength/2.






def calculate_pathloss(total_distance: Union[float, np.ndarray], isLOS: bool, wallExists=False, ignore_shadow_factor=False)->Union[np.ndarray, float]:
    # """
    #
    # Calculate the attenuation of an outdoor link using the 5G path loss model (the close-in free space reference distance
    # model with frequency-dependent path loss exponent). Using equation (5) from [Basar 2020]. OUTPUT IN WATT.
    #
    # :param total_distance: The distance(s) of the total path between the two endpoints of the link. Can be either a numpy array or a float. Note for scattered paths that the total length of the arc is expected.
    # :param isLOS: Whether the link is Line of Sight or not
    # :param wallExists: If a wall interferes in the LOS path. If True, a 10dB penetration loss is induced.
    # :return: The attenuation(s) along the path. The result is either a float or a numpy array depending on the type of `total_distance`.
    # """
    # if isLOS:
    #     n     = n_LOS
    #     f0    = f0_LOS
    #     sigma = sigma_LOS
    #     b     = b_LOS
    #
    # else:
    #     n     = n_NLOS
    #     f0    = f0_NLOS
    #     sigma = sigma_NLOS
    #     b     = b_NLOS
    #
    #
    #
    #
    #
    # if b != 0:
    #     frequency_dependent_term = b *( frequency - f0) / f0
    # else:
    #     frequency_dependent_term = 0
    #
    # pathloss = -20*log10(4*pi/wavelength) - 10*n * (1 + frequency_dependent_term) * safe_log10(total_distance)    # Equation (5)
    # if not ignore_shadow_factor:
    #     shape   = total_distance.shape if isinstance(total_distance, np.ndarray) else [1]  # A trick for the code to work both for arrays and scalar distance
    #     X_sigma = sigma * np.random.rand(*shape)  # Shadow fading term in logarithmic units
    #     pathloss -= X_sigma
    #
    #
    # if pathloss_db_sign == 'positive':
    #     pathloss = -pathloss
    # elif pathloss_db_sign == 'negative':
    #     pass
    # else:
    #     raise ValueError
    #
    # if isinstance(total_distance, np.ndarray):
    #     pathloss = pathloss[0]
    #
    # if wallExists:
    #     pathloss -= wall_attenuation
    #
    # if units_scale == 'power':
    #     pathloss = dBW_to_Watt(pathloss)
    #
    # return pathloss
    return np.power(wavelength/(4*np.pi*total_distance),  2)




def calculate_element_radiation(theta: Union[float, np.ndarray])->Union[float, np.ndarray]:
    """
    Calculate the RIS element radiation for an incoming signal using the cos^q pattern.
    Equation (4) from [Basar 2020]
    :param theta: Elevation angle between the ray and the RIS broadside
    :return: RIS radiation, G_e(theta).
    """
    assert np.all(-pi/2 < theta) and np.all(theta < pi/2)
    if normalize_Ge:
        if isinstance(theta, np.ndarray):
            return np.ones_like(theta)
        else:
            return 1
    else:
        return 2 * (2*q+1) * power(cos(theta), 2*q)









def calculate_array_response(
                             N              : int,
                             phi            : Union[np.ndarray, float],
                             theta          : Union[np.ndarray, float],
                             element_spacing: float,
                            )->ComplexArray:
    """

    Calculate the rotationally symmetric RIS radiation pattern for each element.
    Using equation (3) from [Basar 2020].
    An RIS with uniformly distributed elements is assummed which is positioned either on the xz or the yz planes.

    :param N: The number of RIS elements. Must be a perfect square.
    :param phi: The azimuth angle(s) of arrival between the TX or cluster and the RIS broadside
    :param theta: The elevation angle(s) of arrival between the TX or cluster and the RIS broadside
    :param element_spacing: Used to position elements in a grid
    :return: The array responses for each element. If phi and theta are floats, then it is a 1D array of length N. Otherwise, if phi and theta are numpy arrays, the resulted array has a shape like them but expanded by one axis of size N.
    """

    assert ceil(sqrt(N)) == floor(sqrt(N)), 'N (number of RIS elements) must be a perfect square'
    assert element_spacing > 0            , 'Spacing between elements must be a positive number'
    if isinstance(phi, np.ndarray):
        assert phi.shape == theta.shape     , 'arrays of azimuth and elevation angles must be of equal sizes'

    d                   = element_spacing
    element_coordinates = np.array(list(product(range(int(sqrt(N))), repeat=2))) + 1
    x                   = element_coordinates[:,0]
    y                   = element_coordinates[:,1]

    if isinstance(phi, np.ndarray):
        extended_shape = phi.shape + tuple([len(x)])
        x     = np.broadcast_to(x, extended_shape)
        y     = np.broadcast_to(y, extended_shape)
        theta = theta[..., np.newaxis]
        phi   = phi[..., np.newaxis]

    response = np.exp(1j * k * d * (x * sin(theta) + y * sin(phi) * cos(theta) ) )

    if normalize_steering_vector:
        response = 1/sqrt(N) * response
    return response


def general_channel_equation(l                       : float,
                             dist                    : float,
                             N                       : int=None,
                             theta                   : float=None,
                             phi                     : float=None,
                             element_spacing         : float=None,
                             wallExists              : bool=False,
                             )->Union[complex, ComplexVector]:
    """
    Compute the channel coefficients from a general system model that includes both a LOS and an NLOS (RIS-assisted)
    component. Depending on the value of l, it can model e Rayleigh, a Ricean, or a pure LOS channel.

    :param l: Ratio between LOS and NLOS components. Use larger values for Rayleigh, smaller for Ricean, and 0 for Rayleigh.
    :param dist: The distance (in m) between the transmitting and receiving nodes (to calculate pathloss).
    :param N: Number of RIS elements. Can be ignored if no RIS is involved in the link.
    :param theta: Elevation angle of arrival between the RIS and the transmitter/receiver. Ignored if `N` is None.
    :param phi: Azimuth angle of arrival between the RIS and the transmitter/receiver. Ignored if `N` is None.
    :param element_spacing: Used to position elements in a grid. Ignored if `N` is None.
    :param wallExists: If `True`, induces further pathloss attenuation.
    :return: The channel coefficient(s). If `N` is given, it is an `N`-sized vector, otherwise a float.
    """

    has_NLOS_component = (l != 0)

    if has_NLOS_component:
        Ge             = calculate_element_radiation(theta)
        L_NLOS         = calculate_pathloss(dist, False, wallExists, not shadow_fading_exists)
        NLOS_coeff     = sqrt(Ge * L_NLOS)
        a              = calculate_array_response(N, phi, theta, element_spacing)
        NLOS           = sqrt(l/(l+1)) * NLOS_coeff * a
    else:
        NLOS           = 0

    if not ignore_LOS:
        L_LOS              = calculate_pathloss(dist, True, wallExists, not shadow_fading_exists)
        beta_shape         = N if has_NLOS_component else 1
        beta               = np.random.normal(0, 1, size=beta_shape) + 1j * np.random.normal(0, 1, size=beta_shape)
        LOS                = sqrt(1/(l+1)) * sqrt(L_LOS/2) * beta
    else:
        LOS                = 0


    return LOS + NLOS



def TX_RIS_channel_model(
        N                     : int,
        dist_TX_RIS           : float,
        theta_RIS             : float,
        phi_RIS               : float,
        RIS_element_spacing   : float,
        )->ComplexVector:

    """
    Calculate the TX-RIS channel (h) for a single RIS. This is equation (8) of [Basar 2020].

    The channel is modeled as  Ricean distributed.

    :param N                     Number of RIS elements (which is assumed to be a perfect square).
    :param dist_TX_RIS:          The Line of Sight distance between the TX and the RIS' broadside
    :param theta_RIS:            A vector of length N where the n-th element is the elevation angle of arrival between the TX and the RIS
    :param phi_RIS:              A vector of length N where the n-th element is the azimuth angle of arrival between the TX and the RIS
    :param RIS_element_spacing:  The distance between consecutive RIS elements (assumed to be the same both vertically and horizontally)

    :return:                     The RIS-RX channel coefficients, g, - a complex vector of size N
    """

    h = general_channel_equation(l_h, dist_TX_RIS, N, theta_RIS, phi_RIS, RIS_element_spacing, wallExists=False)
    assert len(h) == N
    return h




def RIS_RX_channel_model(
        N                     : int,
        dist_RIS_RX           : float,
        theta_RIS_RX          : float,
        phi_RIS_RX            : float,
        RIS_element_spacing   : float,
    )->ComplexVector:

    """
    Calculate the RIS-RX channel (h) for a single RIS. This is equation (8) of [Basar 2020].

    The channel is modeled as a clear LOS link without NLOS components in between.
    That is, the assumption that the RX and the RIS are sufficiently close.
    This channel is modeled as a Rayleigh faded indoor channel at 5G frequencies.

    :param N                     Number of RIS elements (which is assumed to be a perfect square).
    :param dist_RIS_RX:          The Line of Sight distance between the TX and the RIS' broadside
    :param theta_RIS_RX:         A vector of length N where the n-th element is the elevation angle of arrival between the RIS and the RX
    :param phi_RIS_RX:           A vector of length N where the n-th element is the azimuth angle of arrival between the RIS and the RX
    :param RIS_element_spacing:  The distance between consecutive RIS elements (assumed to be the same both vertically and horizontally)

    :return:                     The RIS-RX channel coefficients, g, - a complex vector of size N
    """

    g = general_channel_equation(l_g, dist_RIS_RX, N, theta_RIS_RX, phi_RIS_RX, RIS_element_spacing, wallExists=False)
    assert len(g) == N
    return g




def TX_RX_channel_model(TX_RX_distance, wall_exists):
    h_SISO = general_channel_equation(l_SISO, TX_RX_distance, N=None, wallExists=wall_exists)

    h_SISO *= TX_RX_mult_factor

    return h_SISO







def calculate_H(ris_list: List[RIS], TX_location):

    K = sum([ris.total_elements for ris in ris_list])
    H = []

    for i, ris in enumerate(ris_list):
        dist_TX_RIS              = np.linalg.norm(TX_location - ris.position)
        theta_RIS_TX, phi_RIS_TX = ray_to_elevation_azimuth(TX_location, ris.position)
        h                        = TX_RIS_channel_model(ris.total_elements, dist_TX_RIS, theta_RIS_TX,
                                                        phi_RIS_TX, element_spacing )
        H.append(h)

    H = np.array(H).reshape((K, 1))
    return H





def calculate_G_and_h0(ris_list: List[RIS],
                       TX_location,
                      RX_location,):

    TX_location = TX_location.reshape(1, 3)
    K           = sum([ris.total_elements for ris in ris_list])
    G           = []


    for i, ris in enumerate(ris_list):

        dist_RIS_RX              = np.linalg.norm(RX_location - ris.position)
        theta_RIS_RX, phi_RIS_RX = ray_to_elevation_azimuth(ris.position, RX_location)
        g                        = RIS_RX_channel_model(ris.total_elements, dist_RIS_RX, theta_RIS_RX, phi_RIS_RX,
                                                        element_spacing)
        G.append(g)

    TX_RX_distance = np.linalg.norm(TX_location - RX_location)
    h_SISO         = TX_RX_channel_model(TX_RX_distance, wall_exists=True)

    G              = np.array(G).reshape(1, K)
    h0             = np.array(h_SISO).reshape((1, 1))

    return G, h0


