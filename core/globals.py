import numpy as np

random_seed = 1


# Path Loss
referenceDistance       = 1.0  # m
pathlossCoefficientLOS  = 37.5 # dB
pathlossExponentLOS     = 2.2
pathlossCoefficientNLOS = 35.1 # dB
pathlossExponentNLOS    = 3.67






# RIS
total_elements       = (8,8) # prefer square
dependent_elements   = (4,1)  # us a perfect square
total_surfaces       = 4
#elements_per_surface = total_elements/total_surfaces





# EM Waves
carrier_frequency     = 32*10**9 # Hz
light_speed           = 299792458 # m/s
wavelength            = light_speed/float(carrier_frequency) #m
element_width         = wavelength/2.0
mult_fact             = 1  #(element_width^2/wavelength)^2; # is this TX power??

dependents_gap        = wavelength
independent_gap       = wavelength


noisePower   = 100 # dBm
isCorrelated = 0


TX_location = np.array([0, 30, 2])
RX_location = np.array([20,30, 1])

ris_locations = np.array([[15, 25, 2],
                          [15, 35, 2],
                          [25, 25, 2],
                          [25, 35, 2]])

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