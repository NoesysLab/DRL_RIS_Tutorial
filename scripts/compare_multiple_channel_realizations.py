import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core.simulation import Simulator
from utils.data_handlers import SimulationDataset, DataSaver

if len(sys.argv) < 2: raise RuntimeError("Expected setup configuration filename as first argument.")



configuration_filename = sys.argv[1]
sim                    = Simulator(configuration_filename)
dataset                = SimulationDataset(sim.num_RIS, sim.total_RIS_elements, sim.total_RIS_controllable_elements)
dataSaver              = DataSaver(sim.setup_name, './data/simulations/').set_configuration(sim.config)
output_file            = dataSaver.get_save_filename('single_RX_position_many_realizations')




if '--generate' in sys.argv:


    H_prev, G_prev, h0_prev = None, None, None

    for i in tqdm(range(10)):

        H, G, h0           = sim.simulate_transmission(sim.center_RX_position)
        configuration, snr = sim.find_best_configuration(H, G, h0)

        if sim.verbosity >= 3:
            tqdm.write("Best configuration: {} | SNR: {}".format(configuration, snr))



        H_prev, G_prev, h0_prev = np.copy(H), np.copy(G), np.copy(h0)

        dataset.add_datapoint(H, G, h0, sim.RX_locations[i, :], configuration, snr)

    dataset.save(output_file)
    print('Saved to: {}'.format(output_file))


elif '--compare' in sys.argv:


    dataset = dataset.load(output_file+".npy")


    df_phases = pd.DataFrame(data=dataset.get('best_configuration'),
                             columns=['phase_value_{}'.format(i+1) for i in range(sim.total_RIS_controllable_elements)])



    for i in range(sim.total_RIS_controllable_elements):
        p = dataset.get('best_configuration')[:,i].mean()
        print("Mean phase for element {:2d}: {:.2f} ± {:.2f}".format(i,p, p*(1-p)))



    num_evaluations = dataset.shape[0]
    fig, ax = plt.subplots(figsize=(10, 7))
    xs = np.array(range(sim.total_RIS_controllable_elements))
    width = 0.35
    rects1 = ax.bar(xs - width / 2, np.sum(dataset.get('best_configuration'), axis=0), width, label=r'$\pi$')
    rects2 = ax.bar(xs + width / 2, num_evaluations- np.sum(dataset.get('best_configuration'), axis=0), width, label='0')
    ax.set_ylabel('State frequency')
    ax.set_title('Individual element states over multiple evaluations')
    ax.set_xticks(xs)
    ax.set_xticklabels(["Element {}".format(i+1) for i in xs], rotation = 45, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.show()

    try:
        plt.figure()
        sns.histplot(dataset.get('best_SNR'), kde=True, stat='probability', bins=40)
        plt.show()
    except np.linalg.LinAlgError:
        pass

    plt.figure()
    plt.imshow(np.transpose(df_phases.values))
    plt.show()



    # import plotly.graph_objects as go
    #
    # fig = go.Figure(data=
    # go.Parcoords(
    #     line=dict(color=dataset.get('best_SNR'),
    #               colorscale='Electric',
    #               showscale=True,
    #               autocolorscale=True,
    #               ),
    #     dimensions=list([
    #                 dict(range=[0, 1],
    #                     #constraintrange=[0, 1],
    #                     #label='{}'.format(i+1),
    #                     values=dataset.get('best_configuration')[:,i])
    #         for i in range(total_RIS_controllable_elements)
    #
    #         ] +
    #         [    dict(range=[dataset.get('best_SNR').min(), dataset.get('best_SNR').max()],
    #              label='SNR', values=dataset.get('best_SNR')),
    #         ]),
    # )
    # )
    #
    # import dash
    # import dash_core_components as dcc
    # import dash_html_components as html
    # import subprocess
    #
    # result = subprocess.run(['hostname', '-I'], stdout=subprocess.PIPE)
    # local_ip = result.stdout.decode('UTF8').split(' ')[0]
    #
    # # fig.layout.height = 1000
    #
    # app = dash.Dash()
    # app.layout = html.Div([
    #     dcc.Graph(figure=fig, style={'height': '100vh'})
    # ])
    #
    # app.run_server(host=local_ip, port=34533, debug=True, use_reloader=False)
    #

