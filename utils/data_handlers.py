import numpy as np
import os, sys
import pickle
from pathlib import Path
import hashlib
import yaml
from configparser import ConfigParser
from collections import OrderedDict
import json

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
            out = out[:,0:K] + 1j * out[:,K:]

        return out











# def custom_hash(dict_: dict):
#     h = hashlib.blake2b(digest_size=6, salt=b'123')
#     for key, value in dict_.items():
#         h.update("{}:{}".format(key, value).encode('utf8'))
#     return h.hexdigest()

def custom_hash(data: str):
    h = hashlib.blake2b(digest_size=12, salt=b'123')
    h.update(data.encode('utf8'))
    return h.hexdigest()



class DataSaver:
    """
    A class that keeps set_configuration of the experiment configurations and saves user-defined data in separate directories automatically.
    It can be conditioned on any (hashable) objects - i.e. configuration which is saved alongside the rest of the files.
    Each configuration is stored in a separate directory named using the configuration's hash
    It can save and load custom objects to pickle or it provides the appropriate path for custom filenames to be saved independently (e.g. from matplotlib, tensorflow, e.t.c)
    """
    def __init__(self, experiment_name: str, save_dir='./data', ):
        """
        :param experiment_name: A general name for the experiment
        :param save_dir: Top level directory to construct a configuration subdir and save inside
        """
        if '.' in experiment_name:
            experiment_name = '.'.join(experiment_name.split('.')[:-1])

        self.experiment_name = experiment_name
        self.save_dir        = save_dir
        self.configuration   = None # type: ConfigParser
        self._hash           = None

    def _get_experiment_dir(self):
        """
        :return: The exact directory in which data will be stored for this configuration
        """
        if self._hash is None:
            raise ValueError("Not tracking any objects yet.")
        return os.path.join(self.save_dir, self.experiment_name, str(self._hash))

    def _store_configuration_file(self):
        """
        Save a YAML containing the configuration information
        :return:
        """
        dirname = self._get_experiment_dir()
        Path(dirname).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(dirname, self.experiment_name+".ini")
        with open(filename, 'w') as fout:
            self.configuration.write(fout)


    def set_configuration(self, config: ConfigParser, ignore_sections=('program_options', 'constants')):
        """
        Formulate a configuration with any custom objects. This defines a unique directory in which data will be stored
        :return: A reference to the object for chaining
        """
        self.configuration = config
        config_sections = set(config.sections()) - set(ignore_sections)
        sections_data   = [config._sections[section] for section in config_sections]
        data            = {k: v for d in sections_data for k, v in d.items()}
        #data            = OrderedDict(sorted(data.items()))
        data            = str(json.dumps(data, sort_keys=True))
        # data = 'wiiwdhfqiwudgqiywfguiyqcbuyyvcv82t``'
        # print(data)
        self._hash      = custom_hash(data)
        self._store_configuration_file()
        return self

    def store_misc_data(self, data, filename, nested_dir=None):
        """
        Save a serializable object using pickle
        :param data: The object to be stored
        :param filename: The filename (without path)
        :param nested_dir: Potential inner directories that will be made
        :return:
        """

        filename = self.get_save_filename(filename, nested_dir)
        with open(filename, 'wb') as fout:
            pickle.dump(data, fout)

    def load_misc_data(self, filename, nested_dir=None):
        """
        Load a previously stored file for this configuration
        :param filename: Must not include path
        :param nested_dir: Potential inner directories in which the file exists
        :return: Any type of object
        """
        filename = self.get_save_filename(filename, nested_dir)
        with open(filename, 'rb') as fin:
            data     = pickle.load(fin)
            return data

    def store_dict(self, dictionary, filename="results.yml"):
        """
        Store human-readable results in a text file
        :return:
        """
        filename = self.get_save_filename(filename)
        try:
            with open(filename, 'r') as yamlfile:
                cur_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)
                if cur_yaml:
                    cur_yaml.update(dictionary)
        except FileNotFoundError:
            cur_yaml = None

        if cur_yaml:
            with open(filename, 'w') as yamlfile:
                yaml.dump(cur_yaml, yamlfile)
        else:
            with open(filename, 'w') as yamlfile:
                yaml.dump(dictionary, yamlfile)


    def get_save_filename(self, filename, nested_dir=None):
        """
        Construct the exact pathname. Create directories if do not exist
        :param filename:
        :param nested_dir:
        :return:
        """
        dirname = self._get_experiment_dir()
        if nested_dir:
            dirname = os.path.join(dirname, nested_dir)
        Path(dirname).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(dirname, filename)
        return filename


    def __str__(self):
        s = "Setup {} with hash: {}".format(self.experiment_name, self._hash)
        return s
