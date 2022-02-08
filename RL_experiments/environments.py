import itertools
import os
import sys
from typing import Any, Tuple, Union, Type

from tqdm import tqdm

from RL_experiments.standalone_simulatiion import Setup, compute_sum_rate, compute_SINR_per_user, simulate_transmission, \
    construct_precoding_matrix, get_BS_UEs_AoDs

sys.path.insert(0,'..')

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import random_tf_policy
from tf_agents.environments.tf_py_environment import TFPyEnvironment



from utils.custom_configparser import CustomConfigParser
from core import channels
from utils.misc import ray_to_elevation_azimuth, append_to_tuple


class RISEnv2(py_environment.PyEnvironment):


    def get_state(self) -> Any:
        raise NotImplemented

    def set_state(self, state: Any) -> None:
        raise NotImplemented

    def __init__(self, setup: Setup, episode_length, observation_type: str):
        super(RISEnv2, self).__init__()
        self.setup = setup

        self.episode_length      = episode_length
        self._t                  = None
        self._episode_ended      = False
        self.observation_type    = observation_type


        self.observation_noise_variance = setup.observation_noise_variance

        self.num_RIS_configurations      = None
        self.codebook_size_bits_required = None

        self._state = None
        self._curr_reward = 0

        action_dim, observation_dim = self._calculate_space_dimensions()

        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int32,
                                                        minimum=0,
                                                        maximum=action_dim - 1,
                                                        name='action')

        self._observation_spec = array_spec.ArraySpec(shape=(observation_dim,),
                                                      dtype=np.float32,
                                                      name='observation')

        ################# K ###################
        self._info = {}



    @property
    def state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._state

    def _calculate_space_dimensions(self):

        M = self.setup.M
        B = self.setup.B
        K = self.setup.K
        N = self.setup.N
        #N_tot = self.setup.N_tot
        N_controllable = self.setup.N_controllable



        self.num_RIS_configurations      = int(np.power(2, N_controllable))
        self.codebook_size_bits_required = self.setup.K * int(np.ceil(np.log2(self.setup.codebook_rays_per_RX)))
        action_dim                       = int(np.power(2, N_controllable + self.codebook_size_bits_required))



        if self.observation_type == 'channels':

            observation_dim = ((M * B * N) + (M * K * N) + (K * B)) *      2
            #                   +-------+    +---------+   +------+  +---------+
            #                       H             G           h       real+imag

        elif self.observation_type == 'angles':
            observation_dim = 2*self.setup.K  # 2 AoD angles (azimuth and elevation) per RX (AoDs between Bs and UE)

        else:
            raise ValueError("Environment supports only 'channels', 'angles' as `observation_type`.")

        return action_dim, observation_dim







    def generate_observation_vector(self, H, G, h):

        if self.observation_type == 'channels':
            H   = H.flatten()
            G   = G.flatten()
            h   = h.flatten()
            obs = np.concatenate([H.real, H.imag, G.real, G.imag, h.real, h.imag])
            obs = obs.astype(np.float32).flatten()

            if self.observation_noise_variance != 0:
                obs += np.random.normal(loc=0, scale=self.observation_noise_variance, size=len(obs))

            return obs

        elif self.observation_type == 'angles':

            AoDs = get_BS_UEs_AoDs(self.setup).flatten()
            return AoDs.astype('float32')

        else:
            raise ValueError

    def compute_reward(self, H, G, h, RIS_profiles, W):
        SINR = compute_SINR_per_user(self.setup,
                                     H, G, h,
                                     RIS_profiles,
                                     W)
       # return float(np.sum(SINR))
        return compute_sum_rate(SINR)

    def action_2_configuration(self, a):
        bits = np.array([int(x) for x in list('{0:0b}'.format(a))], dtype=int)
        rem_padding = self.setup.N_controllable - len(bits)
        zeros = np.zeros((rem_padding,), dtype=int)
        configuration = np.concatenate([zeros, bits])
        return configuration


    def extend_configuration_to_groups(self, configuration: np.ndarray):
        assert configuration.ndim == 1
        assert len(configuration) == self.setup.N_controllable

        configuration  = np.repeat(configuration, self.setup.group_size)

        assert len(configuration) == self.setup.N_tot

        return configuration

    def configuration2phases(self, configuration):
        # thetas = configuration*np.pi # convert {0,1} to {0, pi}
        # Phi = np.exp(1j*thetas)
        Phi = 2 * configuration - 1  # exp(j*theta) for theta in {0, pi} results in {-1,1}. So better to do it manually for numerical stability
        Phi = Phi.reshape((self.setup.M, self.setup.N))
        return Phi


    def split_action_to_precoding_and_RIS_profile(self, action):
        bits                   = np.array([int(x) for x in list('{0:0b}'.format(action))], dtype=int)
        rem_padding            = self.codebook_size_bits_required + self.setup.N_controllable - len(bits)
        zeros                  = np.zeros((rem_padding,), dtype=int)
        whole_action_bits      = np.concatenate([zeros, bits])
        precoding_bits         = whole_action_bits[:self.codebook_size_bits_required]
        RIS_configuration_bits = whole_action_bits[self.codebook_size_bits_required:]


        return precoding_bits, RIS_configuration_bits



    def select_beamforming_matrix(self, codebook_selection: np.ndarray):
        # codebook: (4, B, K)
        binary_array_to_int    = lambda b: int(b.dot(2**np.arange(b.size)[::-1]))
        bits_per_RX            = self.codebook_size_bits_required // self.setup.K

        W = np.empty(shape=(self.setup.B, self.setup.K), dtype=complex)
        for k in range(self.setup.K):
            this_user_codebook_bits = codebook_selection[k*bits_per_RX:(k+1)*bits_per_RX]
            i                       = binary_array_to_int(this_user_codebook_bits)
            W[:,k]                  = self.setup.codebook[i, :, k]

        return W


    def seed(self, seed_=None):
        pass

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


    def current_time_step(self) -> ts.TimeStep:
        if self.state is not None:
            H, G, h   = self.state
            step_type = np.array(0, dtype='float32')
            obs       = self.generate_observation_vector(H, G, h)
            reward    = np.array(self._curr_reward, dtype='float32')
            discount  = np.array(1, dtype='float32')
            return ts.TimeStep(step_type, reward, discount, obs)
        else:
            return None

    def _reset(self):
        self._t = 0
        self._episode_ended = False
        H, G, h = simulate_transmission(self.setup)
        self._state = (H, G, h)
        obs = self.generate_observation_vector(H, G, h)

        ################# K ###################
        self._info = {}

        return ts.restart(obs)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        codebook_selection, configuration = self.split_action_to_precoding_and_RIS_profile(action)



        #configuration = self.action_2_configuration(action)
        configuration = self.extend_configuration_to_groups(configuration)
        Phi           = self.configuration2phases(configuration)
        W             = self.select_beamforming_matrix(codebook_selection)
        H, G, h       = self._state
        reward        = self.compute_reward(H, G, h, Phi, W)
        H, G, h       = simulate_transmission(self.setup)
        observation   = self.generate_observation_vector(H, G, h)
        self._state   = (H, G, h)
        self._curr_reward = reward
        self._t      += 1
        ################# K ###################
        self._info    = {'t': self._t-1, 'configuration': configuration, 'W': W}


        if self.episode_length is not None and self._t >= self.episode_length:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward, discount=1.0)


    def get_info(self) -> Any:
        ################# K ###################
        return self._info


# -------------------------------------------------------------



class RISEnv3(RISEnv2):

    def __init__(self, setup: Setup, observation_type:str, num_realizations: int, dirname: str, overwrite=False):
        assert num_realizations > 0
        super(RISEnv3, self).__init__(setup, np.inf, observation_type)

        self.num_realizations          = num_realizations
        self.save_dirname              = dirname
        self.save_dirname_H            = os.path.join(self.save_dirname, 'H.npy')
        self.save_dirname_G            = os.path.join(self.save_dirname, 'G.npy')
        self.save_dirname_h            = os.path.join(self.save_dirname, 'h.npy')
        #self.save_dirname_SINR         = os.path.join(self.save_dirname, 'SINR.npy')
        self.save_dirname_BS_RX_angles = os.path.join(self.save_dirname, 'BS_RX_AoDs.npy')

        self.H_all                     = None # type: np.ndarray
        self.G_all                     = None # type: np.ndarray
        self.h_all                     = None # type: np.ndarray
        self.AoDs_all                  = None # type: np.ndarray


        if not os.path.exists(self.save_dirname):
            os.mkdir(self.save_dirname)

        some_files_are_missing = False
        for fname in [self.save_dirname_H, self.save_dirname_G, self.save_dirname_h, self.save_dirname_BS_RX_angles]:
            if not os.path.exists(fname):
                some_files_are_missing = True
                break


        if overwrite or some_files_are_missing or num_realizations is None:
            self._generate_simulation_data()
            self._save_simulation_data()
        else:
            self._load_simulation_data()




    def _generate_simulation_data(self):

        dummy_action = 0
        _ = super()._reset()
        H, G, h = self.state

        H_all    = np.empty(shape=append_to_tuple(H.shape, self.num_realizations, pos=0), dtype=complex)
        G_all    = np.empty(shape=append_to_tuple(G.shape, self.num_realizations, pos=0), dtype=complex)
        h_all    = np.empty(shape=append_to_tuple(h.shape, self.num_realizations, pos=0), dtype=complex)
        AoDs_all = np.zeros(shape=(self.num_realizations, 2*self.setup.K))

        H_all[0, ...]    = H
        G_all[0, ...]    = G
        h_all[0, ...]    = h
        AoDs_all[0, ...] = get_BS_UEs_AoDs(self.setup).flatten()

        pbar = tqdm(range(1, self.num_realizations))
        pbar.set_description('Generating Simulation data')
        for i in pbar:
            _               = super()._step(dummy_action)
            H, G, h         = self.state
            H_all[i,   ...] = H
            G_all[i,   ...] = G
            h_all[i,   ...] = h
            AoDs_all[i,...] = get_BS_UEs_AoDs(self.setup).flatten()



        self.H_all    = H_all
        self.G_all    = G_all
        self.h_all    = h_all
        self.AoDs_all = AoDs_all


    def _save_simulation_data(self):

        if self.H_all is None or self.G_all is None or self.h_all is None or self.AoDs_all is None:
            raise ValueError("Simulation data have not been generated")

        os.makedirs(self.save_dirname, exist_ok=True)

        np.save(self.save_dirname_H, self.H_all)
        np.save(self.save_dirname_G, self.G_all)
        np.save(self.save_dirname_h, self.h_all)
        np.save(self.save_dirname_BS_RX_angles, self.AoDs_all)

        print(f"Simulation data saved under {self.save_dirname}/ .")


    def _load_simulation_data(self):
        self.H_all = np.load(self.save_dirname_H)
        self.G_all = np.load(self.save_dirname_G)
        self.h_all = np.load(self.save_dirname_h)
        self.AoDs_all = np.load(self.save_dirname_BS_RX_angles)

        num_realizations_in_files = np.array(list(map(lambda a: a.shape[0], [self.H_all, self.G_all, self.h_all, self.AoDs_all])))

        if np.any(num_realizations_in_files - num_realizations_in_files[0]):
            raise ValueError("Loaded data with different number of realizations in each channel matrix!")

        if num_realizations_in_files[0] == 0:
            raise ValueError("Loaded empty data!")

        self.num_realizations = num_realizations_in_files[0]
        print(f"Loaded {self.num_realizations} simulation realizations from {self.save_dirname}/ .")


    def _load_curr_realization(self, include_AoDs=False)->\
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

        t = self._t % self.num_realizations
        H = self.H_all[t, ...]
        G = self.G_all[t, ...]
        h = self.h_all[t, ...]

        if not include_AoDs:
            return H, G, h
        else:
            AoDs = self.AoDs_all[t, ...]
            return H, G, h, AoDs

    def _reset(self):
        self._t = 0
        self._episode_ended = False
        H, G, h = self._load_curr_realization()
        self._state = (H, G, h)
        obs = self.generate_observation_vector(H, G, h)
        return ts.restart(obs)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        codebook_selection, configuration = self.split_action_to_precoding_and_RIS_profile(action)

        # configuration = self.action_2_configuration(action)
        configuration     = self.extend_configuration_to_groups(configuration)
        Phi               = self.configuration2phases(configuration)
        W                 = self.select_beamforming_matrix(codebook_selection)
        H, G, h           = self._state
        reward            = self.compute_reward(H, G, h, Phi, W)
        self._t          += 1
        H, G, h           = self._load_curr_realization()
        observation       = self.generate_observation_vector(H, G, h)
        self._state       = (H, G, h)


        if self.episode_length is not None and self._t >= self.episode_length:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward, discount=1.0)













################# K ###################
class RateRequestsEnv(RISEnv3):

    def compute_reward(self, H, G, h, RIS_profiles, W):
        SINR = compute_SINR_per_user(self.setup,
                                     H, G, h,
                                     RIS_profiles,
                                     W)
        requests_penalty = np.clip(SINR-self.setup.rate_requests, a_min=None, a_max=0).sum()
        self._info['SINR'] = SINR
        return requests_penalty



################# K ###################
class RateQoSEnv(RateRequestsEnv):

    def compute_reward(self, H, G, h, RIS_profiles, W):
        requests_penalty = super().compute_reward(H, G, h, RIS_profiles, W)

        SINR = self._info['SINR']
        self._info['requests_penalties'] = np.clip(SINR-self.setup.rate_requests, a_min=None, a_max=0)

        if requests_penalty >= 0:
            return compute_sum_rate(SINR)
        else:
            return requests_penalty
















def evaluate_action(env: RISEnv2, a: int):
    (H, G, h) = env._state
    codebook_selection, configuration = env.split_action_to_precoding_and_RIS_profile(a)
    configuration                     = env.extend_configuration_to_groups(configuration)
    Phi                               = env.configuration2phases(configuration)
    W                                 = env.select_beamforming_matrix(codebook_selection)
    r                                 = env.compute_reward(H, G, h, Phi, W)
    return r

def find_best_action_exhaustively(env: RISEnv2):
    r_max = -float('inf')
    a_best = -1
    for a in range(env.action_spec().maximum):
        r = evaluate_action(env, a)
        if r > r_max:
            r_max, a_best = r, a

    return r_max, a_best

def compute_average_optimal_policy_return(env: RISEnv2, timesteps=1000):

    total_return = 0
    _ = env._reset()

    for _ in tqdm(range(timesteps)):
        r_max, a_best = find_best_action_exhaustively(env)
        total_return += r_max
        _ = env._step(a_best)

    return total_return / timesteps







def get_environment_class_by_type(simulation_type: str)->Type:
    if simulation_type.upper() == 'QOS':
        return RateQoSEnv
    elif simulation_type.upper() == 'REQUESTS':
        return RateRequestsEnv
    else:
        return RISEnv3



















# -------------------------------------------------------------------
class BanditWrapper:
    def __init__(self, tf_env):
        self.env = tf_env

    def generate_reward(self, action):
        time_step = self.env._step(action)
        return time_step.reward.numpy()[0]








if __name__ == '__main__':
    setup = Setup.load_from_json("./parameters.json")
    env = RISEnv2(setup, 1)

    score = compute_average_optimal_policy_return(env, 1000)
    print(f"Optimal policy avg return: {score}")