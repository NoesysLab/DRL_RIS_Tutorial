from typing import List, Union

import numpy as np

from utils.misc import normalize_array, Matrix3D




def digitize_coords(coords: np.array, num: int)->np.array:
    if coords.max() == coords.min():
        return np.zeros_like(coords)
    else:
        dig_coords = (normalize_array(coords)*num).round().astype(int)
        return dig_coords



class PositionGrid:

    RIS_SYMBOL   = '*'
    TX_SYMBOL    = 'x'
    RX_SYMBOL    = 'o'
    EMPTY_SYMBOL = '.'


    def __init__(self, RIS_positions: Matrix3D, TX_positions: Matrix3D, RX_positions: Matrix3D):
        self.RIS_positions = RIS_positions
        self.TX_positions  = TX_positions
        self.RX_positions  = RX_positions

        self.num_RIS       = self.RIS_positions.shape[0]
        self.num_TX        = self.TX_positions.shape[0]
        self.num_RX        = self.RX_positions.shape[0]

        if self.RIS_positions.shape[1] != 3: raise ValueError
        if self.TX_positions.shape[1]  != 3: raise ValueError
        if self.RX_positions.shape[1]  != 3: raise ValueError


    def rotate(self, rotation_matrix):
        raise NotImplementedError


    def shift(self, shift_vector):
        raise NotImplementedError


    def plot_ascii(self, width_chars=20, height_chars=7, return_str=False):
        all_positions = self.TX_positions
        all_positions = np.vstack([all_positions, self.TX_positions, self.RX_positions])
        symbols       = [self.RIS_SYMBOL]*self.num_RIS + \
                        [self.TX_SYMBOL]*self.num_TX + \
                        [self.RX_SYMBOL]*self.num_RX


        Xs = digitize_coords(all_positions[:,0], width_chars-1)
        Ys = digitize_coords(all_positions[:,1], height_chars-1)


        out = np.full((height_chars, width_chars), self.EMPTY_SYMBOL, dtype='U1')
        for x,y,s in zip(Xs,Ys, symbols):
            out[y,x] = s

        s = ''
        for line in out:
            s += "".join(line) + '\n'

        if return_str:
            return s
        else:
            print(s)




    @staticmethod
    def from_ascii(ascii_grid: str,
                   ris_heights: Union[float, List[float]],
                   tx_heights: Union[float, List[float]],
                   rx_heights: Union[float, List[float]],
                   scale_x=None,
                   scale_y=None):

        num_ris = ascii_grid.count(PositionGrid.RIS_SYMBOL)
        num_tx  = ascii_grid.count(PositionGrid.TX_SYMBOL)
        num_rx  = ascii_grid.count(PositionGrid.RX_SYMBOL)


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

                if line[x] == PositionGrid.RIS_SYMBOL:
                    ris_coord_list[num_ris_placed, 0] = actual_x
                    ris_coord_list[num_ris_placed, 1] = actual_y
                    num_ris_placed += 1

                elif line[x] == PositionGrid.TX_SYMBOL:
                    tx_coord_list[num_tx_placed, 0] = actual_x
                    tx_coord_list[num_tx_placed, 1] = actual_y
                    num_tx_placed += 1

                elif line[x] == PositionGrid.RX_SYMBOL:
                    rx_coord_list[num_rx_placed, 0] = actual_x
                    rx_coord_list[num_rx_placed, 1] = actual_y
                    num_rx_placed += 1

                elif line[x] == PositionGrid.EMPTY_SYMBOL:
                    pass
                else:
                    if line[x].isspace():
                        pass
                    else:
                        raise ValueError("Unrecognised character {}".format(line[x]))

        ris_coord_list[:,2] = ris_heights
        tx_coord_list[:,2] = tx_heights
        rx_coord_list[:,2] = rx_heights


        return PositionGrid(ris_coord_list, tx_coord_list, rx_coord_list)




