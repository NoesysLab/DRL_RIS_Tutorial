import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d, Axes3D
import re

from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
from matplotlib import cm
from scipy.interpolate import griddata

from core.geometry import *
from core.surfaces import *
from utils.misc import Matrix2D

grid_plot_params = {
    'show_RIS_ids'   : False,
    'scale'          : 50,
    'color_by_height': True,
    'RIS_color'      : 'black',
    'RIS_symbol'     : 's', # square
    'TX_color'       : 'blue',
    'TX_symbol'      : 'D', # diamond
    'RX_color'       : 'green',
    'RX_symbol'      : 'o', # circle
    'xlims'          : None, # tuple of (min,max)
    'ylims'          : None, # tuple of (min, max)
    'zlims'          : None, # tuple of (min, max)
    'grid'           : True,
    'element_color'  : 'black',
    'alpha'          : 0.7,
    '3D'             : False,
}











def _fix_height_legend(handles, labels, heights, params):
    for i in range(len(handles)):
        handles[i]._marker = MarkerStyle('o', 'full')

    label_numbers = np.array(list(map(lambda s: float(re.sub("[^0-9]", "", s)), labels)))
    label_numbers = (label_numbers-params['min_markersize'])/params['scale']
    #label_numbers = reverse_normalize(label_numbers, heights.min(), heights.max())
    labels = list(map(lambda n: '$\\mathdefault{'+str(n)+"}$", label_numbers))
    return handles, labels

def mscatter(x,y, z=None, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()

    if z is None:
        sc = ax.scatter(x,y,**kw)
    else:
        sc = ax.scatter(x,y,z, **kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc








def plot_positions(RIS_positions, TX_positions, RX_positions, ax=None, params: DefaultDict=None):
    if params is None: params = grid_plot_params

    if ax is None:
        if params['3D']:
            matplotlib.use('TkAgg')
            fig = plt.figure()
            ax = Axes3D(fig)
        else:
            fig, ax = plt.subplots()

    num_RIS = RIS_positions.shape[0]
    num_TX = TX_positions.shape[0]
    num_RX = RX_positions.shape[0]

    all_coords = np.vstack([RIS_positions, TX_positions, RX_positions])
    labels     = [params['RIS_symbol']]*num_RIS + [params['TX_symbol']]*num_TX + [params['RX_symbol']]*num_RX
    sizes      = [params['scale']]*all_coords.shape[0]


    if params['color_by_height'] and not params['3D']:
        params['RIS_color'] = 'k'
        params['TX_color'] = 'k'
        params['RX_color'] = 'k'
        colors = all_coords[:,2]
    else:
        colors = [params['RIS_color']] * num_RIS + \
                 [params['TX_color']] * num_TX + \
                 [params['RX_color']] * num_RX



    if params['3D']:
        scatter = mscatter(all_coords[:,0], all_coords[:,1], all_coords[:,2], ax=ax,
                           s=sizes, c=colors, m=labels)
    else:
        scatter = mscatter(all_coords[:,0], all_coords[:,1], ax=ax,
                           s=sizes, c=colors, m=labels)


    legend_elements = [
        Line2D([0], [0], color=params['RIS_color'], marker=params['RIS_symbol'], linestyle='', label='RIS'),
        Line2D([0], [0], color=params['TX_color'],  marker=params['TX_symbol'], linestyle='', label='TX'),
        Line2D([0], [0], color=params['RX_color'],  marker=params['RX_symbol'],  linestyle='', label='RX')]

    legend1 = ax.legend(handles=legend_elements, loc='best', framealpha=0.8)

    ax.add_artist(legend1)


    if params['color_by_height'] and not params['3D']:
        cbar = plt.colorbar(scatter)
        cbar.ax.set_ylabel(r'$z \ (m)$', rotation=90)

    ax.set_xlabel(r'$x \ (m)$')
    ax.set_ylabel(r'$y \ (m)$')

    if params['3D']:
        ax.set_zlabel(r'$z \ (m)$', rotation=90)

    if params['xlims']: ax.set_xlim(params['xlims'])
    if params['ylims']: ax.set_ylim(params['ylims'])

    if params['grid']: plt.grid()
    return ax







def plot_setup_3D(ris_list: List[RIS], TX_positions, RX_positions, ax=None, params=None, scatterers_positions=None):
    if params is None: params = grid_plot_params

    if ax is None:
        matplotlib.use('TkAgg')
        fig = plt.figure()
        ax = Axes3D(fig)


    all_RIS_element_positions = np.vstack([ris.get_element_coordinates() for ris in ris_list])

    num_RIS_elements = all_RIS_element_positions.shape[0]
    num_TX           = TX_positions.shape[0]
    num_RX           = RX_positions.shape[0]

    all_coords = np.vstack([all_RIS_element_positions, TX_positions, RX_positions])
    labels     = [params['RIS_symbol']]*num_RIS_elements + [params['TX_symbol']]*num_TX + [params['RX_symbol']]*num_RX
    sizes      = [params['scale']]*all_coords.shape[0]


    if params['color_by_height']:
        params['RIS_color'] = 'k'
        params['TX_color'] = 'k'
        params['RX_color'] = 'k'
        colors = all_coords[:,2]
    else:
        colors = [params['RIS_color']] * num_RIS_elements + \
                 [params['TX_color']] * num_TX + \
                 [params['RX_color']] * num_RX




    scatter = mscatter(all_coords[:,0], all_coords[:,1], all_coords[:,2], ax=ax,
                       s=sizes, c=colors, m=labels)



    legend_elements = [
        Line2D([0], [0], color=params['RIS_color'], marker=params['RIS_symbol'], linestyle='', label='RIS'),
        Line2D([0], [0], color=params['TX_color'],  marker=params['TX_symbol'], linestyle='', label='TX'),
        Line2D([0], [0], color=params['RX_color'],  marker=params['RX_symbol'],  linestyle='', label='RX')]

    legend1 = ax.legend(handles=legend_elements, loc='best', framealpha=0.8)

    ax.add_artist(legend1)



    if scatterers_positions is not None:
        mscatter(scatterers_positions[:, 0], scatterers_positions[:, 1], scatterers_positions[:, 2], m='x')



    ax.set_xlabel(r'$x \ (m)$')
    ax.set_ylabel(r'$y \ (m)$')
    ax.set_zlabel(r'$z \ (m)$')

    if params['xlims']: ax.set_xlim(params['xlims'])
    if params['ylims']: ax.set_ylim(params['ylims'])
    if params['zlims']: ax.set_zlim(params['zlims'])

    if params['grid']: plt.grid()
    return ax







def plot_element_matrix(ris: RIS, params=None, ax=None):


    if params is None: params = grid_plot_params





    element_coordinates = ris.get_element_coordinates()


    width               = ris.element_dimensions[0]
    height              = ris.element_dimensions[1]

    rects = []
    for i in range(element_coordinates.shape[0]):
            coords = element_coordinates[i, 0:2]
            rects.append(
                Rectangle(coords, width, height)
            )

    pc = PatchCollection(rects, facecolors=params['element_color'], alpha=params['alpha'])

    if not ax:
        fig, ax = plt.subplots()

    ax.add_collection(pc)

    if params['xlims']: ax.set_xlim(params['xlims'])
    if params['ylims']: ax.set_ylim(params['ylims'])
    return ax




def plot_ris_phase(phase2D: Matrix2D, params=None, ax=None):

    if not ax:
        fig, ax = plt.subplots()



    im = ax.imshow(phase2D)
    plt.colorbar(im)

    width = phase2D.shape[1]
    height = phase2D.shape[0]

    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)




def plot_RIS_3D(ris: RIS, TX_pos: Vector3D, RX_pos: Vector3D, ax=None):
    if ax is None:
        matplotlib.use('TkAgg')
        fig = plt.figure()
        ax = Axes3D(fig)

    ax.scatter(*TX_pos)
    ax.scatter(*RX_pos)

    POS = ris.get_element_coordinates()
    scatter = mscatter(POS[:, 0], POS[:, 1], POS[:, 2], ax=ax, marker='s')
    ax.set_xlabel(r'$x \ (m)$')
    ax.set_ylabel(r'$y \ (m)$')
    ax.set_zlabel(r'$z \ (m)$')
    ax.set_zlim([0,3])









def plot_simulation(RIS_list, cluster_positions, TX_coordinates, center_RX_position):
    scatterers_positions = cluster_positions.reshape((-1, 3))  # Shape (C*Smax, 3)
    scatterers_positions = scatterers_positions[np.all(scatterers_positions != 0, axis=1)]

    params = grid_plot_params.copy()
    params['zlims'] = [0, 3]
    params['color_by_height'] = False
    plot_setup_3D(RIS_list, TX_coordinates.reshape((1, 3)), center_RX_position.reshape(1, 3), params=params,
                  scatterers_positions=scatterers_positions)
    plot_positions(np.array([ris.position for ris in RIS_list]), TX_coordinates.reshape((1, 3)),
                   center_RX_position.reshape(1, 3), )
    plt.show()






def coordinates_heatmap(Xs, Ys, color, cbar_label=None, title=None, x_label='x', y_label='y', smooth=False):
    interpolate = 'linear' if smooth == True else 'nearest'
    gridsize = Xs.shape[0]
    x_min = Xs.min()
    x_max = Xs.max()
    y_min = Ys.min()
    y_max = Ys.max()
    xx = np.linspace(x_min, x_max, gridsize)
    yy = np.linspace(y_min, y_max, gridsize)
    grid = np.array(np.meshgrid(xx, yy.T))
    grid = grid.reshape(2, grid.shape[1] * grid.shape[2]).T
    points = np.array([Xs, Ys]).T  # because griddata wants it that way
    z_grid2 = griddata(points, color, grid, method=interpolate)
    # you get a 1D vector as result. Reshape to picture format!
    z_grid2 = z_grid2.reshape(xx.shape[0], yy.shape[0])
    fig, ax1 = plt.subplots()
    sc = ax1.imshow(z_grid2, extent=[x_min, x_max, y_min, y_max, ],
                    origin='lower', cmap=cm.viridis)
    cbar = plt.colorbar(sc)
    if cbar_label: cbar.ax.set_ylabel(cbar_label, rotation=90)
    if title: plt.title(title)
    if x_label: ax1.set_xlabel(x_label)
    if y_label: ax1.set_ylabel(y_label)


if __name__ == '__main__':


    # # # # # # # # # # # #
    # #  ELEMENT MATRIX   #
    # # # # # # # # # # # #
    #
    # r = RIS([0,0,0], (6,6), (3,2), [1,1], [1,1], [2,2], ('discrete', {'values':[1, np.pi]}))
    #
    # grid_plot_params['xlims'] = (-1, 13)
    # grid_plot_params['ylims'] = (-1, 13)
    #
    # plot_element_matrix(r, grid_plot_params)
    # plt.show()
    #
    #
    #
    #
    # ########################################################
    #
    #
    # # # # # # # # # # # # # #
    # #     POSITION GRID     #
    # # # # # # # # # # # # # #
    #
    # RIS_positions, TX_positions, RX_positions = from_ascii('''
    # ........*......*...o
    # ....................
    # ....................
    # x..................o
    # ....................
    # ....................
    # ..x.*......*.......o
    # ''',
    # 0,1,2,
    # scale_x=3.7,
    # scale_y=4.5)
    #
    #
    # grid_plot_params['2D'] = True
    # grid_plot_params['xlims'] = None
    # grid_plot_params['ylims'] = None
    # plot_positions(RIS_positions, TX_positions, RX_positions)
    # plt.show(block=False)
    #
    #
    # #############################################################
    #
    # # # # # # # # # # # # # #
    # #   RIS PHASE           #
    # # # # # # # # # # # # # #
    #
    # grid_shape = np.array((30,30))
    # group_shape = np.array((3,2))
    # ris = RIS([0,0,0], (12,12), (3,2), [1,1], [1,1], [2,2], ('discrete', {'values':[1, np.pi]}))
    #
    # ris.set_random_state()
    #
    # plot_ris_phase(ris.get_phase('2D'))
    # plt.show(block=True)



    # # # # # # # # # # # # # # #
    # #      RIS 3D             #
    # # # # # # # # # # # # # # #
    # TX_pos = np.array([1, 1, 1])
    # RX_pos = np.array([5, 5, 1])
    # ris = RIS([4,2,1], RX_pos - TX_pos,   (6,6), (3,2), [0.1,0.1], [0.01,0.01], [0.02,0.02], ('discrete', {'values':[1, np.pi]}))
    #
    # params = grid_plot_params.copy()
    # params['zlims'] = [0,2]
    # params['color_by_height'] = False
    #
    # plot_RIS_3D(ris, TX_pos, RX_pos)
    # plt.show()




    # # # # # # # # # # # # # #
    #      Setup 3D           #
    # # # # # # # # # # # # # #
    TX_pos = np.array([0, 0, 1])
    RX_pos = np.array([1, 1, 1])
    r1 = RIS([0.5, 0, 1], RX_pos - TX_pos, (6, 6), (3, 2), [0.001, 0.01], [0.01, 0.01], [0.02, 0.02],
              ('discrete', {'values': [1, np.pi]}))

    r2 = RIS([1, 0.5, 1], RX_pos - TX_pos, (6, 6), (3, 2), [0.001, 0.01], [0.01, 0.01], [0.02, 0.02],
               ('discrete', {'values': [1, np.pi]}))

    r3 = RIS([0.5, 1, 1], RX_pos - TX_pos, (6, 6), (3, 2), [0.001, 0.01], [0.01, 0.01], [0.02, 0.02],
               ('discrete', {'values': [1, np.pi]}))

    r4 = RIS([0, 0.5, 1], RX_pos - TX_pos, (6, 6), (3, 2), [0.001, 0.01], [0.01, 0.01], [0.02, 0.02],
               ('discrete', {'values': [1, np.pi]}))


    params = grid_plot_params.copy()
    params['zlims'] = [0,2]
    params['color_by_height'] = False

    plot_setup_3D([r1,r2,r3,r4], TX_pos.reshape((1,3)), RX_pos.reshape((1,3)), params=params)
    plt.show()













    pass