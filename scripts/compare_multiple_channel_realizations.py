from datetime import datetime
import sys
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core.channels import compute_SNR
from core.setup import load_config_from_file, create_setup_from_config
from core.simulation import exhaustive_search, calculate_H, calculate_G_and_h0, generate_clusters, calculate_RX_scatterers_distances, initialize_from_config
from core.surfaces import RIS
from utils.data_handlers import SimulationDataset, DataSaver
from utils.misc import diag_per_row
from utils.plotting import plot_simulation


if len(sys.argv) < 2:
    raise RuntimeError("Expected setup configuration filename as first argument.")
else:
    configuration_filename = sys.argv[1]
    SETUP = configuration_filename.split("/")[-1].split(".")[0]






config    = load_config_from_file(configuration_filename)
dataSaver = DataSaver(SETUP, config.get('program_options', 'output_directory_root')).set_configuration(config)

print("Using setup '{}'. ".format(SETUP))


batch_size      = config.getint('program_options', 'batch_size')
verbosity       = config.getint('program_options', 'verbosity_level')
seed            = config.getint('program_options', 'random_seed')
stop_iterations = config.getint('program_options', 'stop_after_evaluations')




if seed:
    np.random.seed(seed)
    random.seed(seed)


initialize_from_config(config)

[RIS_list,
 RX_locations,
 TX_coordinates,
 RIS_coordinates,
 lambda_p,
 num_clusters,
 num_RIS,
 total_RIS_elements,
 total_RIS_controllable_elements,
 transmit_power,
 noise_power,
 center_RX_position              ] = create_setup_from_config(config)

[Sc,
 cluster_positions,
 TX_clusters_distances,
 clusters_RIS_distances,
 thetas_AoA,
 phis_AoA                         ] = generate_clusters(TX_coordinates,
                                                        RIS_coordinates,
                                                        lambda_p,
                                                        num_clusters)



ris = RIS_list[0] # type: RIS










def configuration2phases(configurations: np.ndarray):
    assert configurations.ndim == 2
    dependent_elements_per_RIS = ris.num_dependent_elements

    phase_space  = ris.phase_space
    phases       = phase_space.calculate_phase_shifts(configurations)
    phases       = np.repeat(phases, repeats=dependent_elements_per_RIS, axis=1)
    Phi          = diag_per_row(phases)
    return Phi




dataset = SimulationDataset(num_RIS, total_RIS_elements, total_RIS_controllable_elements)
output_file = dataSaver.get_save_filename('single_RX_position_many_realizations')




if '--generate' in sys.argv:


    for i in tqdm(range(100)):

        H                     = calculate_H(RIS_list, TX_coordinates, Sc, TX_clusters_distances,
                                            clusters_RIS_distances, thetas_AoA, phis_AoA)
        G, h0                 = calculate_G_and_h0(RIS_list, TX_coordinates, center_RX_position)
        configuration, snr    = exhaustive_search(RIS_list, H, G, h0, noise_power, batch_size=batch_size, show_progress_bar=False)

        if verbosity >= 3:
            tqdm.write("Best configuration: {} | SNR: {}".format(configuration, snr))

        dataset.add_datapoint(H, G, h0, RX_locations[i, :], configuration, snr)

    dataset.save(output_file)
    print('Saved to: {}'.format(output_file))


elif '--compare' in sys.argv:


    dataset = dataset.load(output_file+".npy")


    # df = pd.DataFrame({
    #     'H_real': dataset.get('H').real,
    #     'H_imag': dataset.get('H').imag,
    #     'H_mag' : np.absolute(dataset.get('H')),
    #     'H_angle': np.angle(dataset.get('H')),
    #     'G': dataset.get('G'),
    #     'h0': dataset.get('h'),
    #     'best_snr' : dataset.get('best_SNR'),
    # })

    df_phases = pd.DataFrame(data=dataset.get('best_configuration'),
                             columns=['phase_value_{}'.format(i+1) for i in range(total_RIS_controllable_elements)])
    #df_phases['SNR'] = dataset.get('best_SNR')



    for i in range(total_RIS_controllable_elements):
        print("Average phase for element {:2d}: {:.2f} ± {:.2f}".format(i,
                                                                      dataset.get('best_configuration')[:,i].mean(),
                                                                      dataset.get('best_configuration')[:,i].std(),))

    fig = plt.figure(figsize=(10, 7))

    bp = plt.boxplot(dataset.get('best_configuration'), showmeans=True)

    plt.figure()
    sns.histplot(dataset.get('best_SNR'), kde=True, stat='probability', bins=40)
    plt.show()

    plt.figure()
    plt.imshow(np.transpose(df_phases.values))
    plt.show()

    # plt.figure()
    # plt.imshow(np.random.binomial(n=1, p=0.5, size=(16,100)))
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(np.random.binomial(n=1, p=0.5, size=(16, 100)))
    # plt.show()


    # plt.figure()
    # df_phases.sum().plot.bar()
    # plt.show()

    import plotly.graph_objects as go

    fig = go.Figure(data=
    go.Parcoords(
        line=dict(color=dataset.get('best_SNR'),
                  colorscale='Electric',
                  showscale=True,
                  autocolorscale=True,
                  ),
        dimensions=list([
                    dict(range=[0, 1],
                        #constraintrange=[0, 1],
                        #label='{}'.format(i+1),
                        values=dataset.get('best_configuration')[:,i])
            for i in range(total_RIS_controllable_elements)

            ] +
            [    dict(range=[dataset.get('best_SNR').min(), dataset.get('best_SNR').max()],
                 label='SNR', values=dataset.get('best_SNR')),
            ]),
    )
    )

    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    import subprocess

    result = subprocess.run(['hostname', '-I'], stdout=subprocess.PIPE)
    local_ip = result.stdout.decode('UTF8').split(' ')[0]

    # fig.layout.height = 1000

    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig, style={'height': '100vh'})
    ])

    app.run_server(host=local_ip, port=34533, debug=True, use_reloader=False)


