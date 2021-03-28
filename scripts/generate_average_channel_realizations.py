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
#dataset                = SimulationDataset(sim.num_RIS, sim.total_RIS_elements, sim.total_RIS_controllable_elements)
#dataSaver              = DataSaver(sim.setup_name)


inp_dataset = SimulationDataset(sim.num_RIS, sim.total_RIS_elements, sim.total_RIS_controllable_elements)\
    .load('data/simulations/setup1/987b6478417007f985e675cd/exhaustive_search.npy')


out_dataset = SimulationDataset(sim.num_RIS, sim.total_RIS_elements, sim.total_RIS_controllable_elements)

num_samples = 400

RX_positions        = inp_dataset.get('RX_position')
best_configurations = inp_dataset.get('best_configuration')
best_SNRs           = inp_dataset.get('best_SNR')


H_samples  = np.empty((num_samples, sim.total_RIS_elements), dtype=complex)
G_samples  = np.empty((num_samples, sim.total_RIS_elements), dtype=complex)
h0_samples = np.empty(num_samples, dtype=complex)


for i in tqdm(range(len(RX_positions))):


    for j in range(num_samples):
        H, G, h0           = sim.simulate_transmission(sim.RX_locations[i,:])

        H_samples[j, :] = H.flatten()
        G_samples[j, :] = G.flatten()
        h0_samples[j]  = h0.flatten()

    H_mean = H_samples.mean(axis=0).flatten()
    G_mean = G_samples.mean(axis=0).flatten()
    h0_mean = G_samples.mean().flatten()

    out_dataset.add_datapoint(H_mean, G_mean, h0_mean, RX_positions[i,:],best_configurations[i,:], best_SNRs[i])


out_dataset.save('data/simulations/setup1/987b6478417007f985e675cd/simulation_mean_channels')
