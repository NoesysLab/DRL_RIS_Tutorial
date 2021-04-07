import sys


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import  scipy as sc

from sklearn.neighbors import NearestNeighbors


from core.simulation import Simulator
from utils.plotting import coordinates_heatmap


def hamming_distance(u, v):
    a = u-v
    b = np.abs(a)
    c = b.sum()
    return int(c)



def hamming_distance_with_neighbors(XY, configurations):
    configurations = configurations.astype(int)
    min_hamming_distances = np.empty(XY.shape[0], dtype=int)
    avg_hamming_distances = np.empty(XY.shape[0], dtype=float)
    NN = NearestNeighbors(n_neighbors=9)
    NN.fit(XY)


    _, closest_points_indices = NN.kneighbors(XY)

    for i in range(XY.shape[0]):
        this_configuration        = configurations[i,:]
        neighbor_configurations   = [configurations[j,:] for j in closest_points_indices[i,:]]
        neighbor_hamming_dists    = [hamming_distance(this_configuration, conf) for conf in neighbor_configurations]
        neighbor_hamming_dists    = sorted(neighbor_hamming_dists)
        min_neighbor_hamming_dist = neighbor_hamming_dists[1]
        min_hamming_distances[i]  = min_neighbor_hamming_dist
        neighbor_hamming_dists    = np.array(neighbor_hamming_dists[1:])
        avg_hamming_distances[i]  = neighbor_hamming_dists.mean()

    return min_hamming_distances, avg_hamming_distances




def visualize(sim: Simulator, mode: str):

    dataset       = sim.get_dataset()
    data_filename = sim.get_exhaustive_results_filename(mode)
    print('Loading from "{}"'.format(data_filename))
    dataset.load(data_filename)

    Xs             = dataset.get('RX_position')[:,0]
    Ys             = dataset.get('RX_position')[:,1]
    SNRs           = dataset.get('best_SNR')[:]
    configurations = dataset.get('best_configuration')
    bits_set       = np.sum(configurations, axis=1)

    min_hamming_dists, avg_hamming_dists = hamming_distance_with_neighbors(dataset.get('RX_position')[:,0:2], configurations)

    coordinates_heatmap(Xs, Ys, SNRs, 'SNR', 'Optimal SNR in every RX position')
    plt.savefig(sim.dataSaver.get_save_filename(mode+'_SNR_locality.png'))
    plt.show(block=False)

    coordinates_heatmap(Xs, Ys, bits_set, 'Number of active elements', 'Optimal phase configuration in every RX position')
    plt.savefig(sim.dataSaver.get_save_filename(mode+'_Active_elements_locality.png'))
    plt.show(block=False)


    coordinates_heatmap(Xs, Ys, min_hamming_dists, 'Minimum Hamming distance with neighbors', 'Local changes in optimal phase configurations')
    plt.savefig(sim.dataSaver.get_save_filename(mode+'_Hamming_locality.png'))
    plt.show(block=False)

    coordinates_heatmap(Xs, Ys, avg_hamming_dists, 'Average Hamming distance with neighbors', 'Local changes in optimal phase configurations')
    plt.savefig(sim.dataSaver.get_save_filename(mode+'_average_Hamming_locality.png'))
    plt.show(block=False)



if __name__ == '__main__':
    if len(sys.argv) < 2: raise RuntimeError("Expected setup configuration filename as first argument.")
    configuration_filename = sys.argv[1]
    sim_ = Simulator(configuration_filename)

    visualize(sim_, 'train')
    visualize(sim_, 'test')