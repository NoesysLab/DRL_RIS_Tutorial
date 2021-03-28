import pandas as pd

from deprecated.setup import load_config_from_file, create_setup_from_config
from utils.data_handlers import SimulationDataset

config = load_config_from_file('setups/setup1.ini')


[_,
 _,
 _,
 _,
 _,
 _,
 num_RIS,
 total_RIS_elements,
 total_RIS_controllable_elements,
 _,
 _,
 _              ] = create_setup_from_config(config)


num_RIS_elements = total_RIS_elements // num_RIS

dataset = SimulationDataset(num_RIS, total_RIS_elements, total_RIS_controllable_elements)


SIMULATION_DIR = 'data/simulations/setup1/9c947d9979970f1e0c1453b3/'
dataset.load(SIMULATION_DIR+'simulation.npy')


K = total_RIS_elements
K1 = total_RIS_controllable_elements

num_columns = 2*K + 2*K + 2 + 3 + K1 + 1


column_names  = ['H_{}_{}_real'.format(i+1,j+1) for i in range(num_RIS) for j in range(num_RIS_elements)]
column_names += ['H_{}_{}_imag'.format(i+1,j+1) for i in range(num_RIS) for j in range(num_RIS_elements)]
column_names += ['G_{}_{}_real'.format(i+1,j+1) for i in range(num_RIS) for j in range(num_RIS_elements)]
column_names += ['G_{}_{}_imag'.format(i+1,j+1) for i in range(num_RIS) for j in range(num_RIS_elements)]
column_names += ['h_SISO_real', 'h_SISO_imag']
column_names += ['RX_x', 'RX_y', 'RX_z']
column_names += ['phase_value_{}'.format(i+1) for i in range(total_RIS_controllable_elements)]
column_names += ['SNR']


print(dataset.values.shape)
print(len(column_names))
print(num_columns)



df = pd.DataFrame(data=dataset.values, columns=column_names)
df.to_csv(SIMULATION_DIR+'simulation.csv', index=False)