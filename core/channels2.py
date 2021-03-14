from numpy import sin, cos, pi, sqrt, sum, floor, ceil, power, log10, exp
import numpy as np
from typing import *
from itertools import product
import warnings

from utils.complex import sample_gaussian_complex_matrix
from utils.custom_types import Vector, Matrix2D, Matrix3D, ComplexVector, ComplexArray, Vector3D, Matrix3DCoordinates
from utils.misc import safe_log10

lightspeed = 299702547         # on air
frequency  = 32*10**9
wavelength = lightspeed/float(frequency)
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







def calculate_pathloss(total_distance: Union[float, np.ndarray], isLOS: bool, wallExists=False)->Union[np.ndarray, float]:
    """

    Calculate the attenuation of an indoor link using the 5G path loss model (the close-in free space reference distance
    model with frequency-dependent path loss exponent). Using equation (5) from [Basar 2020]. Output in dB.

    :param total_distance: The distance(s) of the total path between the two endpoints of the link. Can be either a numpy array or a float. Note for scattered paths that the total length of the arc is expected.
    :param isLOS: Whether the link is Line of Sight or not
    :param wallExists: If a wall interferes in the LOS path. If True, a 10dB penetration loss is induced.
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
    pathloss = -20*log10(4*pi/wavelength) - 10*n * (1 + b *( frequency - f0) / f0) * safe_log10(total_distance) - X_sigma    # Equation (5)


    pathloss = -pathloss


    if not isinstance(total_distance, np.ndarray):
        pathloss = pathloss[0]

    if wallExists:
        pathloss -= 10 # todo: Make it + !!!!

    return pathloss





def calculate_element_radiation(theta: Union[float, np.ndarray])->Union[float, np.ndarray]:
    """
    Calculate the RIS element radiation for an incoming signal using the cos^q pattern.
    Equation (4) from [Basar 2020]
    :param theta: Elevation angle between the ray and the RIS broadside
    :return: RIS radiation, G_e(theta).
    """
    assert np.all(-pi/2 < theta) and np.all(theta < pi/2)
    return 2 * (2*q+1) * power(cos(theta), 2*q)









def calculate_array_response(
                             N              : int,
                             phi            : Union[np.ndarray, float],
                             theta          : Union[np.ndarray, float],
                             element_spacing: float)->ComplexArray:
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
    element_coordinates = np.array(list(product(range(int(sqrt(N))), repeat=2)))
    x                   = element_coordinates[:,0]
    y                   = element_coordinates[:,1]

    if isinstance(phi, np.ndarray):
        extended_shape = phi.shape + tuple([len(x)])
        x     = np.broadcast_to(x, extended_shape)
        y     = np.broadcast_to(y, extended_shape)
        theta = theta[..., np.newaxis]
        phi   = phi[..., np.newaxis]

    response = np.exp(1j * k * d * (x * sin(theta) + y * sin(phi) * cos(theta) ) )
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




def TX_RIS_channel_model(
        N                     : int,
        Sc                    : List[int],
        theta_scatterers      : Matrix2D,
        phi_scatterers        : Matrix2D,
        length_path_scatterers: Matrix2D,
        dist_TX_RIS           : float,
        theta_RIS_LOS         : float,
        phi_RIS_LOS           : float,
        RIS_element_spacing   : float,
        LOS_component_exists  : bool,
        )->ComplexVector:

    """

    Calculate the TX-RIS channel (h) for a single RIS. This is equation (2) of [Basar 2020].

    This channel is modeled as a Ricean faded indoor channel at 5G frequencies.
    For notation, let C be the number of clusters, Sc be the number of scatterers/IOs/rays of the c-th cluster,
    Smax be the maximum number of scatterers in any given cluster.
    The following matrices should be appended with 0 where Sc is smaller than Smax.

    :param N                       Number of RIS elements (which is assumed to be a perfect square).
    :param Sc                      A list of integers of length C, each denoting the number of scatterers in the corresponding cluster
    :param theta_scatterers:       A matrix of shape (C, Smax) where the (c,s)-th element is the elevation angle of arrival between the s-th scatterer of the c-th cluster and the RIS broadside.
    :param phi_scatterers:         A matrix of shape (C, Smax) where the (c,s)-th element is the azimuth angle of arrival between the s-th scatterer of the c-th cluster and the RIS broadside.
    :param length_path_scatterers: A matrix of shape (C, Smax) where the (c,s)-th element is the length of the path from the TX to the RIS broadside after being scattered by the s-th scatterer of the c-th cluster - i.e. d_{c,s} = a_c + b_{c,s}
    :param dist_TX_RIS:            The Line of Sight distance between the TX and the RIS broadside
    :param theta_RIS_LOS:          The elevation angle of arrival between the TX and the RIS
    :param phi_RIS_LOS:            The azimuth angle of arrival between the TX and the RIS
    :param RIS_element_spacing:    The distance between consecutive RIS elements (assumed to be the same both vertically and horizontally)
    :param LOS_component_exists    Whether the channel must account for a LOS effect as well. This is passed as a parameter to allow coupling the random choice with the TX-RX channel.
    :return:                       The TX-RIS channel coefficients, h, - a complex vector of size N
    """


    C      = len(Sc)
    Smax   = theta_scatterers.shape[1]

    assert np.max(Sc) == Smax
    assert theta_scatterers.shape[0] == phi_scatterers.shape[0] == C
    assert theta_scatterers.shape[1] == phi_scatterers.shape[1]


    # Non-LOS component: Summed array response vectors and link attenuations for each sub-ray

    gamma  = sqrt(1/sum(Sc))                                                                     # normalization factor
    bettas = sample_gaussian_complex_matrix((C,Smax))                                            # Complex Array of shape (C,Smax)   - Complex Gaussian distributed path gain of the (c,s)-th scatterer
    Ge     = calculate_element_radiation(theta_scatterers)                                       # Array of shape (C,Smax) - RIS element radiation from (4)
    L      = calculate_pathloss(length_path_scatterers, isLOS=False)                             # Array of shape (C,Smax)   - Attenuation along the (c,s)-th propagation path
    a      = calculate_array_response(N, phi_scatterers, theta_scatterers, RIS_element_spacing)  # Complex Array of shape (C,Smax,N) - Array response of the RIS
    part2D = bettas * sqrt(Ge * L)
    h_NLOS = gamma * sum( part2D[...,np.newaxis] * a, axis=(0,1))                                # Compute the first term of (2) by adding scattered gains, attenuations and responses along all (C,Sc) scatterers.


    # LOS component: Same strategy but for the single TX-RIS path

    if LOS_component_exists:
        Ge_LOS = calculate_element_radiation(theta_RIS_LOS)                                      # float - RIS element radiation from (4)
        L_LOS  = calculate_pathloss(dist_TX_RIS, isLOS=True)                                     # float - attenuation in the LOS link
        eta    = np.random.uniform(0, 2*pi, size=N)                                              # Array of shape (N) - Random phase term
        a_LOS  = calculate_array_response(N, phi_RIS_LOS, theta_RIS_LOS, RIS_element_spacing)    # Complex Array of shape (N) - Array response of the RIS
        h_LOS  = sqrt(Ge_LOS * L_LOS) * exp(1j * eta) * a_LOS                                    # LOS component of the channel coefficients from (6).
    else:
        h_LOS  = 0

    h = h_NLOS + h_LOS

    assert len(h) == N
    return h




def RIS_RX_channel_model(
        N                     : int,
        dist_RIS_RX           : float,
        theta_RIS_RX          : float,
        phi_RIS_RX            : float,
        RIS_element_spacing   : float,
        wall_exists           : bool=False)->ComplexVector:

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

    Ge     = calculate_element_radiation(theta_RIS_RX)                                          # float - RIS element radiation from (4)
    L      = calculate_pathloss(dist_RIS_RX, isLOS=True, wallExists=wall_exists)                                        # float - attenuation across the link path
    eta    = np.random.uniform(0, 2 * pi, size=N)                                               # Array of shape (N) - Random phase term
    a      = calculate_array_response(N, phi_RIS_RX, theta_RIS_RX, RIS_element_spacing)         # Array of shape (N) - Array response of the RIS
    g      = sqrt(Ge * L) * exp(1j * eta) * a                                                   # LOS component of the channel coefficients from (6).

    assert len(g) == N
    return g




def TX_RX_channel_model(TX_RX_distance, wall_exists):
    h0 = sample_gaussian_complex_matrix((1,1)) * sqrt(calculate_pathloss(TX_RX_distance, isLOS=False, wallExists=wall_exists))
    return h0


# def TX_RX_channel_model(
#         Sc                      : List[int],
#         TX_scatterers_distances : Vector,
#         scatterers_RIS_distances: Vector,
#         scatterers_RX_distances : Vector,
#         TX_RX_distance          : float,
#         LOS_component_exists    : bool,
#         wall_exists             : bool,
#         )->ComplexVector:
#
#     """
#
#     Calculate the RX-RX channel using the SISO mmWave channel modeling (Equation (9) of [Basar 2020]).
#
#     :param Sc:                        A list of integers of length C, each denoting the number of scatterers in the corresponding cluster
#     :param TX_scatterers_distances:   A matrix of size C, with the c-th element being the distance between the TX and the c-th cluster.
#     :param scatterers_RIS_distances:  A matrix of size C, with the c-th element being the distance between the c-th cluster and the RIS.
#     :param scatterers_RX_distances:   A matrix of size C, with the c-th element being the distance between the c-th cluster and the RX.
#     :param TX_RX_distance:            The distance between the TX and the RX
#     :param LOS_component_exists:      Whether the channel must account for a LOS effect as well. This is passed as a parameter to allow coupling the random choice with the TX-RIS channel.
#     :return:                          h_SISO: The complex scalar channel gain of the TX-RX link
#     """
#
#
#     C     = len(Sc)
#     Smax  = np.max(Sc)
#
#     assert len(TX_scatterers_distances) == len(scatterers_RIS_distances) == len(scatterers_RX_distances) == C
#
#     # Non-LOS component: Summed link attenuations for each cluster sub-ray
#
#     gamma                = sqrt(1 / sum(Sc))                                                                 # normalization factor
#     bettas               = sample_gaussian_complex_matrix((C, Smax))                                         # Complex Array of shape (C,Smax)   - Complex Gaussian distributed path gain of the (c,s)-th scatterer
#     total_link_distances = TX_scatterers_distances + scatterers_RX_distances
#     L                    = calculate_pathloss(total_link_distances, isLOS=False, wallExists=wall_exists)
#     eta_epsilon          = k * (scatterers_RIS_distances - scatterers_RX_distances)
#     h_SISO_NLOS          = gamma * sum(bettas * exp(1j * eta_epsilon) * sqrt(L), axis=(0,1))
#
#
#     # LOS component: Pathloss in the TX-RX path
#     # todo: Should that take the RIS into consideration???????
#     if LOS_component_exists:
#         L          = calculate_pathloss(TX_RX_distance, isLOS=True)
#         eta        = np.random.uniform(0, 2 * pi)
#         h_SISO_LOS = sqrt(L)*exp(1j*eta)
#     else:
#         h_SISO_LOS = 0
#
#     h_SISO = h_SISO_NLOS + h_SISO_LOS
#     return h_SISO




def _generate_scatterers_positions(C, Sc, Smax, TX_coordinates, RIS_Coordinates, phi_TX, theta_TX):
    y_bounds = [np.min(RIS_Coordinates[:, 1])+0.01, np.max(RIS_Coordinates[:, 1])-0.01]
    x_bounds = [TX_coordinates[0]+0.01, np.min(RIS_Coordinates[:, 0])-0.01]
    z_bounds = [0, TX_coordinates[2]-0.01]

    bounds = np.array([x_bounds, y_bounds, z_bounds])

    cluster_positions = np.zeros((C, Smax, 3))

    min_TX_RIS_dist = np.min(np.linalg.norm(TX_coordinates - RIS_Coordinates, axis=1))


    for c in range(C):

        cluster_centroid_coords = np.random.uniform(low=bounds[:,0], high=bounds[:,1])

        for s in range(Sc[c]):

            rotation_matrix     = [cos(theta_TX[c][s])*cos(phi_TX[c][s]), cos(theta_TX[c][s])*sin(phi_TX[c][s]), sin(theta_TX[c][s])]
            scatterer_positions = np.array(rotation_matrix) * np.random.rand(3) + cluster_centroid_coords
            scatterer_positions = np.clip(scatterer_positions, a_min=bounds[:,0], a_max=bounds[:,1])


            cluster_positions[c, s, :] = scatterer_positions #[x, y, z]



    return cluster_positions



def _calculate_RIS_scatterers_distances_and_angles(C, Sc, Smax, RIS_Coordinates, cluster_positions):
    num_RIS = RIS_Coordinates.shape[0]

    clusters_RIS_distances = np.zeros((C, Smax, num_RIS))
    thetas_AoA             = np.zeros((C, Smax, num_RIS))
    phis_AoA               = np.zeros((C, Smax, num_RIS))

    for r in range(num_RIS):

        x_RIS = RIS_Coordinates[r,0]
        y_RIS = RIS_Coordinates[r,1]
        z_RIS = RIS_Coordinates[r,2]

        for c in range(C):
            for s in range(Sc[c]):
                x,y,z                         = cluster_positions[c,s,:]
                b_c_s                         = np.linalg.norm(RIS_Coordinates[r, :] - cluster_positions[c,s,:])
                clusters_RIS_distances[c,s,r] = b_c_s
                thetas_AoA[c,s,r]             = np.sign(z - z_RIS) * np.arcsin( np.abs(z_RIS - z) / b_c_s )
                phis_AoA[c,s,r]               = np.sign(x_RIS - x) * np.arctan( np.abs(x_RIS - x) / np.abs(y_RIS - y) )

    return clusters_RIS_distances, thetas_AoA, phis_AoA



def calculate_RX_scatterers_distances(Sc, RX_coordinates, cluster_positions):
    RX_clusters_distances = np.linalg.norm(RX_coordinates[None, None, :] - cluster_positions, axis=2)  # Shape (C, Sc)

    C = len(Sc)
    Smax = np.max(Sc)

    for c in range(C):
        for s in range(Sc[c], Smax):
            RX_clusters_distances[c, s] = 0

    return RX_clusters_distances








def generate_clusters(TX_coordinates : Vector3D,
                      RIS_Coordinates: Matrix3DCoordinates,
                      lambda_p       : float):

    # assuming TX is on the yz plane and all RIS on the xz plane

    C             = np.maximum(2, np.random.poisson(lambda_p))
    Sc            = np.random.randint(1, 30, size=C)
    Smax          = np.max(Sc)

    print("Generating {} clusters with {} scatterers.".format(C, Sc))

    mean_phi_TX   = np.random.uniform(-pi/2, pi/2, size=C)
    mean_theta_TX = np.random.uniform(-pi/4, pi/4, size=C)

    phi_TX        = [np.random.laplace(mean_phi_TX[c]  , 5*pi/180, size=Sc[c]) for c in range(C)]
    theta_TX      = [np.random.laplace(mean_theta_TX[c], 5*pi/180, size=Sc[c]) for c in range(C)]


    cluster_positions = _generate_scatterers_positions(C, Sc, Smax, TX_coordinates, RIS_Coordinates, phi_TX, theta_TX)

    TX_clusters_distances = np.linalg.norm(TX_coordinates[None,None,:]-cluster_positions, axis=2) # Shape (C, Sc)

    for c in range(C):
        for s in range(Sc[c], Smax):
            TX_clusters_distances[c,s] = 0


    clusters_RIS_distances,\
    thetas_AoA,\
    phis_AoA = _calculate_RIS_scatterers_distances_and_angles(C, Sc, Smax, RIS_Coordinates, cluster_positions)

    return Sc, cluster_positions, TX_clusters_distances, clusters_RIS_distances, thetas_AoA, phis_AoA


