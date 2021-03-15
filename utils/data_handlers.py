import numpy as np


class SimulationDataset:
    def __init__(self,
                 num_RIS,
                 total_RIS_elements,
                 total_RIS_configurable_elements,
                 num_datapoints=None):

        self.num_RIS                         = num_RIS
        self.total_RIS_elements              = total_RIS_elements
        self.total_RIS_configurable_elements = total_RIS_configurable_elements
        self.dataset_length                  = num_datapoints
        self.__last_row_index                = 0

        K  = total_RIS_elements
        K1 = total_RIS_configurable_elements

        # Row format.
        # Note that H,G,h are complex, so 2 columns are needed for each array element
        # Also note that for H,G,h all K real values are saved first and then all K imaginary values
        # ---------------------------------------
        # H                 : First 2*K columns
        # G                 : Next 2*K columns
        # h                 : Next 2 columns
        # RX_position       : Next 3 columns
        # best_configuration: Next K1 columns
        # best_SNR          : Final column

        array_columns = 2*K + 2*K + 2 + 3 + K1 + 1

        if self.dataset_length is not None:
            self.values = np.empty(shape=(self.dataset_length, array_columns))
        else:
            self.values = list()

        self.array_columns = array_columns



    def check_sizes(self, H, G, h, RX_position, best_configuration, best_SNR):
        sizes_ok = len(H) == self.total_RIS_elements and len(G) == self.total_RIS_elements

        if hasattr(h, '__len__'):
            sizes_ok = sizes_ok and len(h) == 1
        elif isinstance(h, complex):
            sizes_ok = sizes_ok and True
        else:
            sizes_ok = False

        sizes_ok = sizes_ok and len(RX_position) == 3
        sizes_ok = sizes_ok and len(best_configuration) == self.total_RIS_configurable_elements

        if hasattr(best_SNR, '__len__'):
            sizes_ok = sizes_ok and len(best_SNR) == 1
        elif isinstance(best_SNR, float):
            sizes_ok = sizes_ok and True
        else:
            sizes_ok = False

        return sizes_ok


    def add_datapoint(self, H, G, h, RX_position, best_configuration, best_SNR):

        H                  = H.flatten()
        G                  = G.flatten()
        h                  = h.flatten()
        RX_position        = RX_position.flatten()
        best_configuration = best_configuration.flatten()
        try:
            best_SNR       = best_SNR.flatten()
        except AttributeError:
            pass

        if not self.check_sizes(H, G, h, RX_position, best_configuration, best_SNR):
            raise ValueError

        row = np.hstack([H.real, H.imag, G.real, G.imag, h.real, h.imag, RX_position, best_configuration, best_SNR])
        row = row.flatten()

        if self.dataset_length is not None:
            if self.__last_row_index < self.dataset_length:
                self.values[self.__last_row_index, :] = row
                self.__last_row_index += 1
            else:
                raise ValueError
        else:
            self.values.append(row)


    def save(self, filename, mode='wb'):

        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values)

        with open(filename, mode=mode) as fout:
            np.save(fout, self.values, allow_pickle=True)



    def load(self, filename, mode='rb'):
        with open(filename, mode=mode) as fin:
            self.values = np.load(fin, allow_pickle=True)
            print('Loaded array of shape: {}'.format(self.values.shape))
            if self.values.shape[1] != self.array_columns:
                raise ValueError

    def get(self, column_name:str):
        if column_name not in {'H', 'G', 'h', 'RX_position', 'best_configuration', 'best_SNR'}:
            raise ValueError

        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values)


        K  = self.total_RIS_elements
        K1 = self.total_RIS_configurable_elements

        if column_name == 'H':
            out = self.values[:, 0:2*K]

        elif column_name == 'G':
            out = self.values[:, 2*K:4*K]

        elif column_name == 'h':
            out = self.values[:, 4*K:4*K+2]

        elif column_name == 'RX_position':
            out = self.values[:, 4*K+2:4*K+5]

        elif column_name == 'best_configuration':
            out = self.values[:, 4*K+5:4*K+5+K1]

        elif column_name == 'best_SNR':
            out = self.values[:, -1]

        else:
            assert False


        if column_name in ['H','G', 'h']:
            out = np.complex(real=out[:,0:K], imag=out[:,K:])


        return out



