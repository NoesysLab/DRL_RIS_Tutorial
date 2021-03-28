from datetime import datetime
import sys
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt




from core.simulation import Simulator
from utils.data_handlers import SimulationDataset, DataSaver


if len(sys.argv) < 2: raise RuntimeError("Expected setup configuration filename as first argument.")

configuration_filename = sys.argv[1]
sim                    = Simulator(configuration_filename)
dataset                = SimulationDataset(sim.num_RIS, sim.total_RIS_elements, sim.total_RIS_controllable_elements)
dataSaver              = DataSaver(sim.setup_name)



RX_positions = dataset.get('RX_position')


H  = dataset.get('H')
G  = dataset.get('G')
h0 = dataset.get('h')




configurations_best = dataset.get('best_configuration')
SNRs_best           = dataset.get('best_SNR')



configurations_random = np.random.randint(0, sim.num_RIS_states, size=configurations_best.shape)
Phi_random            = sim.configuration2phases(configurations_random)
SNRs_random           = sim.calculate_SNR(H, G, Phi_random, h0).flatten()



Xs = np.arange(0, dataset.shape[0])


fig, ax = plt.subplots()
ax.plot(Xs, SNRs_best, label='Exhaustive search')
ax.plot(Xs, SNRs_random, label='Random phases')
plt.legend()

ax.fill_between(Xs, SNRs_best, SNRs_random, alpha=0.5, color='gray')

ax.set_xlabel('RX position')
ax.set_ylabel('SNR (strange units)')
plt.show()


fig, ax = plt.subplots()
colors = SNRs_random - SNRs_best
im = ax.scatter(RX_positions[:,0], RX_positions[:,1], c=colors)
plt.colorbar(im)
plt.show()

