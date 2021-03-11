from numpy import sin, cos, pi, sqrt, sum, floor, ceil, power, log10, exp
import numpy as np
from typing import *
from itertools import product
import warnings

from utils.complex import sample_gaussian_complex_matrix
from utils.custom_types import Vector, Matrix2D, Matrix3D, ComplexVector, ComplexArray



wavelength = 1111111.0         # todo: change here
lightspeed = 299702547         # on air
frequency  = lightspeed/wavelength
k          = 2*pi/wavelength
q          = 0.285

f0_LOS     = 24.2*10**9
n_LOS      = 3.19
b_LOS      = 0.06
sigma_LOS  = 8.29

f0_NLOS    = 24.2*10**9
n_NLOS     = 1.73
b_NLOS     = 0
sigma_NLOS = 3.02







def calculate_pathloss(total_distance: Union[float, np.ndarray], isLOS: bool)->Union[np.ndarray, float]:
    """

    Calculate the attenuation of an indoor link using the 5G path loss model (the close-in free space reference distance
    model with frequency-dependent path loss exponent). Using equation (5) from [Basar 2020]. Output in dB.

    :param total_distance: The distance(s) of the total path between the two endpoints of the link. Can be either a numpy array or a float. Note for scattered paths that the total length of the arc is expected.
    :param isLOS: Whether the link is Line of Sight or not
    :return: The attenuation(s) along the path. The result is either a float or a numpy array depending on the type of `total_distance`.
    """
    if isLOS:
        n     = n_LOS
        f0    = f0_LOS
        sigma = sigma_LOS
        b     = b_LOS

    else:
        n     = n_NLOS
        f0    = f0_NLOS
        sigma = sigma_NLOS
        b     = b_NLOS


    shape = total_distance.shape if isinstance(total_distance, np.ndarray) else [1]                                     # A trick for the code to work both for arrays and scalar distance

    X_sigma  = sigma * np.random.rand(*shape)                                                                           # Shadow fading term in logarithmic units
    pathloss = -20*log10(4*pi/wavelength) - 10*n * (1 + b *( frequency - f0) / f0) * log10(total_distance) - X_sigma    # Equation (5)


    # if isinstance(total_distance, np.ndarray):
    #     return pathloss
    # else:
    #     return pathloss[0]

    return pathloss





def calculate_element_radiation(theta: np.ndarray)->np.ndarray:
    """
    Calculate the RIS element radiation for an incoming signal using the cos^q pattern.
    Equation (4) from [Basar 2020]
    :param theta: Elevation angles between the ray and every element of the RIS broadside
    :return: Element radiations, G_e(theta) - array of the same shape as theta.
    """
    assert np.all(theta > - pi/2) and np.all(theta < pi/2)
    return 2 * (2*q+1) * power(cos(theta), 2*q)









def calculate_array_response(
                             N              : int,
                             phi            : np.ndarray,
                             theta          : np.ndarray,
                             element_spacing: float)->ComplexArray:
    """

    Calculate the rotationally symmetric RIS radiation pattern for each element.
    Using equation (3) from [Basar 2020].
    An RIS with uniformly distributed elements is assummed which is positioned either on the xz or the yz planes.

    :param N: The number of RIS elements. Must be a perfect square.
    :param phi: The azimuth angles of arrival between
    :param theta:
    :param element_spacing:
    :return:
    """

    assert ceil(sqrt(N)) == floor(sqrt(N)), 'N (number of RIS elements) must be a perfect square'
    assert element_spacing > 0            , 'Spacing between elements must be a positive number'
    assert len(phi) == len(theta)         , 'arrays of azimuth and elevation angles must be of equal sizes'

    d                   = element_spacing
    element_coordinates = np.array(list(product(range(int(sqrt(N))), repeat=2)))
    x                   = element_coordinates[:,0]
    y                   = element_coordinates[:,1]
    response            = np.exp(1j * k * d * (x * sin(theta) + y * sin(phi) * cos(theta) ) )

    return response






def LOS_link_exists(dist: float)->bool:
    p = None                                                                                     # Probability of existing a LOS path
    if dist <= 1.2:                                                                              # Equation (7) - LOS probabilities for indoor places
        p = 1
    elif 1.2 < dist <= 6.5:
        p = 1
    else:
        p = 0.32*exp(-( (dist-6.5)/32.6 ) )

    I_h = np.random.binomial(n=1, p=p)                                                           # Sample from a Bernoulli distribution whether a TX-RIS path exists
    return I_h == 1




def calculate_TX_RIS_channel(
        Sc                    : List[int],
        theta_scatterers      : Matrix3D,
        phi_scatterers        : Matrix3D,
        length_path_scatterers: Matrix2D,
        dist_TX_RIS           : float,
        theta_RIS_LOS         : Vector,
        phi_RIS_LOS           : Vector,
        RIS_element_spacing   : float,
        LOS_component_exists  : bool,)->ComplexVector:

    """

    Calculate the TX-RIS channel (h) for a single RIS. This is equation (2) of [Basar 2020].

    This channel is modeled as a Ricean faded indoor channel at 5G frequencies.
    For notation, let C be the number of clusters, Sc be the number of scatterers/IOs/rays of the c-th cluster,
    Smax be the maximum number of scatterers in any given cluster,
    N be the number of RIS elements (which is assumed to be a perfect square).
    The following matrices should be appended with 0 where Sc is smaller than Smax.

    :param Sc                      A list of integers of length C, each denoting the number of scatterers in the corresponding cluster
    :param theta_scatterers:       A matrix of shape (C, Smax, N) where the (c,s,n)-th element is the elevation angle of arrival between the s-th scatterer of the c-th cluster and the n-th RIS element.
    :param phi_scatterers:         A matrix of shape (C, Smax, N) where the (c,s,n)-th element is the azimuth angle of arrival between the s-th scatterer of the c-th cluster and the n-th RIS element.
    :param length_path_scatterers: A matrix of shape (C, Smax) where the (c,s)-th element is the length of the path from the TX to the RIS' broadside after being scattered by the s-th scatterer of the c-th cluster - i.e. d_{c,s} = a_c + b_{c,s}
    :param dist_TX_RIS:            The Line of Sight distance between the TX and the RIS' broadside
    :param theta_RIS_LOS:          A vector of length N where the n-th element is the elevation angle of arrival between the TX and the RIS
    :param phi_RIS_LOS:            A vector of length N where the n-th element is the azimuth angle of arrival between the TX and the RIS
    :param RIS_element_spacing:    The distance between consecutive RIS elements (assumed to be the same both vertically and horizontally)
    :param LOS_component_exists    Whether the channel must account for a LOS effect as well. This is passed as a parameter to allow coupling the random choice with the TX-RX channel.
    :return:                       The TX-RIS channel coefficients, h, - a complex vector of size N
    """


    C      = len(Sc)
    Smax   = theta_scatterers.shape[1]
    N      = theta_scatterers.shape[2]

    assert max(Sc) == Smax
    assert theta_scatterers.shape[0] == phi_scatterers.shape[0] == C
    assert theta_scatterers.shape[1] == phi_scatterers.shape[1]
    assert theta_scatterers.shape[2] == phi_scatterers.shape[2]
    assert len(theta_RIS_LOS) == len(phi_RIS_LOS) == N


    # Non-LOS component: Summed array response vectors and link attenuations for each sub-ray

    gamma  = sqrt(1/sum(Sc))                                                                     # normalization factor
    bettas = sample_gaussian_complex_matrix((C,Smax))                                            # Complex Array of shape (C,Smax)   - Complex Gaussian distributed path gain of the (c,s)-th scatterer
    Ge     = calculate_element_radiation(theta_scatterers)                                       # Array of shape (C,Smax,N) - RIS element radiation from (4)
    L      = calculate_pathloss(length_path_scatterers, isLOS=False)                             # Array of shape (C,Smax)   - Attenuation along the (c,s)-th propagation path
    L      = np.repeat(L[:, :, np.newaxis], repeats=N, axis=2)                                   # Convert L to a 3D matrix of shape (C,Smax,N) by repeating the same path loss for all RIS elements
    a      = calculate_array_response(N, phi_scatterers, theta_scatterers, RIS_element_spacing)  # Complex Array of shape (C,Smax,N) - Array response of the RIS
    h_NLOS = gamma * sum(bettas * sqrt(Ge * L) * a, axis=[0,1])                                  # Compute the first term of (2) by adding scattered gains, attenuations and responses along all (C,Sc) scatterers.


    # LOS component: Same strategy but for the single TX-RIS path

    if LOS_component_exists:
        Ge_LOS = calculate_element_radiation(theta_RIS_LOS)                                      # Array of shape (N) - RIS element radiation from (4)
        L_LOS  = calculate_pathloss(dist_TX_RIS, isLOS=True)                                     # float - attenuation in the LOS link
        eta    = np.random.uniform(0, 2*pi, size=N)                                              # Array of shape (N) - Random phase term
        a_LOS  = calculate_array_response(N, phi_RIS_LOS, theta_RIS_LOS, RIS_element_spacing)    # Complex Array of shape (N) - Array response of the RIS
        h_LOS  = sqrt(L_LOS * Ge_LOS) * exp(1j * eta) * a_LOS                                    # LOS component of the channel coefficients from (6).
    else:
        h_LOS  = 0

    h = h_NLOS + h_LOS
    return h




def calculate_RIS_RX_channel(
        dist_RIS_RX           : Matrix2D,
        theta_RIS_RX          : Vector,
        phi_RIS_RX            : Vector,
        RIS_element_spacing   : float,)->ComplexVector:

    """
    Calculate the RIS-RX channel (h) for a single RIS. This is equation (8) of [Basar 2020].

    The channel is modeled as a clear LOS link without NLOS components in between.
    That is, the assumption that the RX and the RIS are sufficiently close.
    This channel is modeled as a Rayleigh faded indoor channel at 5G frequencies.

    :param dist_RIS_RX:          The Line of Sight distance between the TX and the RIS' broadside
    :param theta_RIS_RX:         A vector of length N where the n-th element is the elevation angle of arrival between the RIS and the RX
    :param phi_RIS_RX:           A vector of length N where the n-th element is the azimuth angle of arrival between the RIS and the RX
    :param RIS_element_spacing:  The distance between consecutive RIS elements (assumed to be the same both vertically and horizontally)

    :return:                     The RIS-RX channel coefficients, g, - a complex vector of size N
    """

    N      = theta_RIS_RX.shape[0]
    Ge     = calculate_element_radiation(theta_RIS_RX)                                          # Array of shape (N) - RIS element radiation from (4)
    L      = calculate_pathloss(dist_RIS_RX, isLOS=True)                                        # float - attenuation across the link path
    eta    = np.random.uniform(0, 2 * pi, size=N)                                               # Array of shape (N) - Random phase term
    a      = calculate_array_response(N, phi_RIS_RX, theta_RIS_RX, RIS_element_spacing)         # Array of shape (N) - Array response of the RIS
    g      = sqrt(L * Ge) * exp(1j * eta) * a                                                   # LOS component of the channel coefficients from (6).

    return g




def calculate_TX_RX_channel(
        Sc                      : List[int],
        TX_scatterers_distances : Vector,
        scatterers_RIS_distances: Vector,
        scatterers_RX_distances : Vector,
        TX_RX_distance          : float,
        LOS_component_exists    : bool,
        )->ComplexVector:

    """

    Calculate the RX-RX channel using the SISO mmWave channel modeling (Equation (9) of [Basar 2020]).

    :param Sc:                        A list of integers of length C, each denoting the number of scatterers in the corresponding cluster
    :param TX_scatterers_distances:   A matrix of size C, with the c-th element being the distance between the TX and the c-th cluster.
    :param scatterers_RIS_distances:  A matrix of size C, with the c-th element being the distance between the c-th cluster and the RIS.
    :param scatterers_RX_distances:   A matrix of size C, with the c-th element being the distance between the c-th cluster and the RX.
    :param TX_RX_distance:            The distance between the TX and the RX
    :param LOS_component_exists:      Whether the channel must account for a LOS effect as well. This is passed as a parameter to allow coupling the random choice with the TX-RIS channel.
    :return:                          h_SISO: The complex scalar channel gain of the TX-RX link
    """


    C     = len(Sc)
    Smax  = max(Sc)

    assert len(TX_scatterers_distances) == len(scatterers_RIS_distances) == len(scatterers_RX_distances) == C

    # Non-LOS component: Summed link attenuations for each cluster sub-ray

    gamma                = sqrt(1 / sum(Sc))                                                                 # normalization factor
    bettas               = sample_gaussian_complex_matrix((C, Smax))                                         # Complex Array of shape (C,Smax)   - Complex Gaussian distributed path gain of the (c,s)-th scatterer
    total_link_distances = TX_scatterers_distances + scatterers_RX_distances
    L                    = calculate_pathloss(total_link_distances, isLOS=False)
    eta_epsilon          = k * (scatterers_RIS_distances - scatterers_RX_distances)
    h_SISO_NLOS          = gamma * sum(bettas * exp(1j * eta_epsilon) * sqrt(L))


    # LOS component: Pathloss in the TX-RX path
    # todo: Should that take the RIS into consideration???????
    if LOS_component_exists:
        L          = calculate_pathloss(TX_RX_distance, isLOS=True)
        eta        = np.random.uniform(0, 2 * pi)
        h_SISO_LOS = sqrt(L)*exp(1j*eta)
    else:
        h_SISO_LOS = 0

    h_SISO = h_SISO_NLOS + h_SISO_LOS
    return h_SISO





