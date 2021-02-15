import numpy as np

random_seed = 1


# Path Loss
referenceDistance       = 1.0  # m
pathlossCoefficientLOS  = 37.5 # dB
pathlossExponentLOS     = 2.2
pathlossCoefficientNLOS = 35.1 # dB
pathlossExponentNLOS    = 3.67






# RIS
total_elements       = 16 # prefer square
dependent_elements   = 9  # us a perfect square
total_surfaces       = 8
elements_per_surface = total_elements/total_surfaces





# EM Waves
carrier_frequency     = 32*10**9 # Hz
light_speed           = 299792458 # m/s
wavelength            = light_speed/float(carrier_frequency) #m
element_width         = wavelength/2.0
mult_fact             = 1  #(element_width^2/wavelength)^2;

dependents_gap        = wavelength
independent_gap       = wavelength


noisePower   = 1
isCorrelated = 0


TX_location = np.array([0, 0, 2])
RX_location = np.array([20,0, 1])


gridRotate = np.array([
    [1,0,0],
    [0,0,0],
    [0,1,0]
])
gridShift = np.array([
    10,
    10,
    2
])