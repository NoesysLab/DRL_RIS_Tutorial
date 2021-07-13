import numpy as np


from core.simulation import Simulator
from core.channels import *



sim = Simulator('./setups/noise_free2.ini')
initialize_from_config(sim.config)


d          = 15.
beta_shape = 10
beta       = np.random.normal(0, 1, size=beta_shape) + 1j * np.random.normal(0, 1, size=beta_shape)

#print(np.linalg.norm(beta))

L = calculate_pathloss(d, True, True, True)

#print(L)

print(np.linalg.norm(sqrt(L/2)*beta))


"""

300859.0524416138  | False, False
951399.8603955422  | False, True
 58455.83542376896 | True, False
184853.58246706394 | True, True



-----------------------------
2.960009806299024e-05      | False, False
9.360372884338737e-06      | False, True
1.523400000000000e-04      | True, False
4.817573636689898e-05      | True, True


"""