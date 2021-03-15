from typing import *
from math import floor, ceil, sqrt
import numpy as np

from utils.misc import normalize_array, Matrix3DCoordinates



RIS_SYMBOL   = 'x'
TX_SYMBOL    = '*'
RX_SYMBOL    = 'o'
EMPTY_SYMBOL = '.'



def digitize_coords(coords: np.array, num: int)->np.array:
    if coords.max() == coords.min():
        return np.zeros_like(coords)
    else:
        dig_coords = (normalize_array(coords)*num).round().astype(int)
        return dig_coords



def apply_transformation(coordinates: Matrix3DCoordinates, rotate_matrix=None, shift_vector=None):

    if not rotate_matrix: rotate_matrix = np.eye(coordinates.shape[0])
    if not shift_vector:  shift_vector  = np.array([0,0,0])

    shift_matrix = np.repeat(shift_vector, coordinates.shape[0]).reshape((coordinates.shape[0], 3))


    new_coordinates = rotate_matrix * coordinates + shift_matrix
    return new_coordinates



def grid_to_ascii(RIS_positions: Matrix3DCoordinates, TX_positions: Matrix3DCoordinates, RX_positions: Matrix3DCoordinates, width_chars=20, height_chars=7, return_str=False):
    all_positions = TX_positions
    all_positions = np.vstack([all_positions, TX_positions, RX_positions])
    symbols       = [RIS_SYMBOL]*RIS_positions.shape[0] + \
                    [TX_SYMBOL]*TX_positions.shape[0] + \
                    [RX_SYMBOL]*RX_positions.shape[0]


    Xs = digitize_coords(all_positions[:,0], width_chars-1)
    Ys = digitize_coords(all_positions[:,1], height_chars-1)


    out = np.full((height_chars, width_chars), EMPTY_SYMBOL, dtype='U1')
    for x,y,s in zip(Xs,Ys, symbols):
        out[y,x] = s

    s = ''
    for line in out:
        s += "".join(line) + '\n'

    if return_str:
        return s
    else:
        print(s)





def from_ascii(ascii_grid: str,
               ris_heights: Union[float, List[float]],
               tx_heights: Union[float, List[float]],
               rx_heights: Union[float, List[float]],
               scale_x=None,
               scale_y=None):

    num_ris = ascii_grid.count(RIS_SYMBOL)
    num_tx  = ascii_grid.count(TX_SYMBOL)
    num_rx  = ascii_grid.count(RX_SYMBOL)


    if not hasattr(ris_heights, '__iter__'): ris_heights = [ris_heights] * num_ris
    if not hasattr(tx_heights, '__iter__'): tx_heights = [tx_heights] * num_tx
    if not hasattr(rx_heights, '__iter__'): rx_heights = [rx_heights] * num_rx

    if len(ris_heights) != num_ris: raise ValueError
    if len(tx_heights) != num_tx: raise ValueError
    if len(rx_heights) != num_rx: raise ValueError



    num_ris_placed = 0
    num_tx_placed  = 0
    num_rx_placed  = 0
    ris_coord_list = np.empty(shape=(num_ris, 3))
    tx_coord_list  = np.empty(shape=(num_tx, 3))
    rx_coord_list  = np.empty(shape=(num_rx, 3))



    ascii_grid = ascii_grid.strip()
    lines = ascii_grid.split('\n')
    for y in range(len(lines)):
        line = lines[y].strip()
        for x in range(len(line)):

            actual_x = x if not scale_x else x*scale_x
            actual_y = y if not scale_y else y*scale_y

            if line[x] == RIS_SYMBOL:
                ris_coord_list[num_ris_placed, 0] = actual_x
                ris_coord_list[num_ris_placed, 1] = actual_y
                num_ris_placed += 1

            elif line[x] == TX_SYMBOL:
                tx_coord_list[num_tx_placed, 0] = actual_x
                tx_coord_list[num_tx_placed, 1] = actual_y
                num_tx_placed += 1

            elif line[x] == RX_SYMBOL:
                rx_coord_list[num_rx_placed, 0] = actual_x
                rx_coord_list[num_rx_placed, 1] = actual_y
                num_rx_placed += 1

            elif line[x] == EMPTY_SYMBOL:
                pass
            else:
                if line[x].isspace():
                    pass
                else:
                    raise ValueError("Unrecognised character {}".format(line[x]))

    ris_coord_list[:,2] = ris_heights
    tx_coord_list[:,2] = tx_heights
    rx_coord_list[:,2] = rx_heights


    return ris_coord_list, tx_coord_list, rx_coord_list





def get_random_2D_positions_on_square(num_positions: int, xy_center: Tuple[float,float], width: float, z_value:Union[float,Iterable]=None)->Matrix3DCoordinates:
    positions        = np.empty(shape=(num_positions, 3))
    positions[:,0:2] = np.random.random((num_positions, 2))
    positions[:,2]   = z_value
    return positions

def get_2D_positions_on_square_grid(num_positions: int, xy_center: Tuple[float,float], width: float, z_value:Union[float,Iterable]=None)->Matrix3DCoordinates:

    num_x_positions     = ceil(sqrt(num_positions))
    num_y_positions     = floor(sqrt(num_positions))

    start_x             = xy_center[0] - width/2
    end_x               = xy_center[0] + width/2
    start_y             = xy_center[1] - width/2
    end_y               = xy_center[1] + width/2

    xs                  = np.linspace(start_x, end_x, num_x_positions)
    ys                  = np.linspace(start_y, end_y, num_y_positions)

    X, Y                = np.meshgrid(xs, ys)

    positions_grid      = np.empty(shape=(num_x_positions*num_y_positions, 3 if z_value is not None else 2))
    positions_grid[:,0] = X.flatten()
    positions_grid[:,1] = Y.flatten()

    if z_value is not None:
        positions_grid[:,2] = z_value

    return positions_grid



def get_receiver_positions(placement_type:str,
                           num_positions: int,
                           xy_center: Tuple[float,float],
                           width: float,
                           z_value:Union[float,Iterable]=None
                           )->Matrix3DCoordinates:
    if placement_type == 'grid':
        return get_2D_positions_on_square_grid(num_positions, xy_center, width, z_value)
    elif placement_type == 'random':
        return get_random_2D_positions_on_square(num_positions, xy_center, width, z_value)
    else:
        raise ValueError("Only supporting 'grid' or 'random' placement types.")





def rotate(xy, ang):
    coords = np.array(xy).reshape((2,1))
    R      = np.array([[np.cos(ang), -np.sin(ang)],
                     [np.sin(ang),  np.cos(ang)]])

    return (R @ coords).reshape(xy.shape)



def rotate_along_coords(xy, xy_base, ang_rad):
    return rotate(xy-xy_base, ang_rad)+xy_base