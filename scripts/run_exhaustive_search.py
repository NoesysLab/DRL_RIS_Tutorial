from datetime import datetime
import sys
from tqdm import tqdm

from core.simulation import Simulator
from utils.misc import fmt_position
from utils.visualize_exhaustive_search import visualize
from utils.custom_types import Matrix3DCoordinates






def run_search(sim: Simulator, RX_locations: Matrix3DCoordinates, mode: str):

    print(f'Running exhaustive search to generate {mode} dataset.')

    start_t = datetime.now()
    dataset  = sim.get_dataset()
    iterator = range(RX_locations.shape[0])

    if sim.verbosity >= 2:
        print("\nExhaustive search:")
        iterator = tqdm(iterator)


    for i in iterator:
        if sim.stop_iterations is not None and i>= sim.stop_iterations: break

        position           = RX_locations[i, :]
        H, G, h0           = sim.simulate_transmission(position)
        configuration, snr = sim.find_best_configuration(H, G, h0)
        dataset.add_datapoint(H, G, h0, position, configuration, snr)

        if sim.verbosity >= 3: tqdm.write("Best configuration for {}: {} | SNR: {}".format(fmt_position(position), configuration, snr))


    filename = sim.get_exhaustive_results_filename(mode)
    dataset.save(filename)



    end_t = datetime.now()
    duration = end_t-start_t
    if sim.verbosity >= 1:
        print(f"Finished. Time elapsed: {duration}")

    print(f"Saved to {filename} .")




if __name__ == '__main__':
    if len(sys.argv) < 2:
        configuration_filename = 'setups/'+input('Insert setup filename: ')+'.ini'
    else:
        configuration_filename = sys.argv[1]

    sim = Simulator(configuration_filename)

    run_search(sim, sim.RX_train_locations, 'train')
    visualize(sim, 'train')

    run_search(sim, sim.RX_test_locations, 'test')
    visualize(sim, 'test')
