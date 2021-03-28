from datetime import datetime
import sys
from tqdm import tqdm

from core.simulation import Simulator
from utils.data_handlers import SimulationDataset, DataSaver


if len(sys.argv) < 2: raise RuntimeError("Expected setup configuration filename as first argument.")



start_t = datetime.now()


configuration_filename = sys.argv[1]
sim                    = Simulator(configuration_filename)
dataset                = SimulationDataset(sim.num_RIS, sim.total_RIS_elements, sim.total_RIS_controllable_elements)
dataSaver              = DataSaver(sim.setup_name)






iterator = range(sim.RX_locations.shape[0])
if sim.verbosity >= 2:
    print("\nExhaustive search:")
    iterator = tqdm(iterator)


for i in iterator:
    if sim.stop_iterations is not None and i>= sim.stop_iterations: break

    H, G, h0           = sim.simulate_transmission(sim.RX_locations[i,:])
    configuration, snr = sim.find_best_configuration(H, G, h0)

    if sim.verbosity >= 3:
        tqdm.write("Best configuration: {} | SNR: {}".format(configuration, snr))

    dataset.add_datapoint(H, G, h0, sim.RX_locations[i, :], configuration, snr)






dataset.save(dataSaver.get_save_filename(sim.config.get('program_options', 'output_file_name')),
             formats=sim.config.getlist('program_options', 'output_extensions', is_numerical=False))



end_t = datetime.now()
duration = end_t-start_t
if sim.verbosity >= 1:
    print("Finished. Time elapsed: {}".format(duration))

print("Saved to {} .".format(dataSaver.get_save_filename('')))