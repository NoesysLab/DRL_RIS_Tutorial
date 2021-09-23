import sys
from typing import Any

from RL_experiments.standalone_simulatiion import Setup, compute_sum_rate, compute_SINR_per_user, simulate_transmission

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
from utils.misc import ray_to_elevation_azimuth



class RIS_TFenv(py_environment.PyEnvironment):
    def __init__(self, config: CustomConfigParser, episode_length=None, transmit_SNR=1):
        super(RIS_TFenv, self).__init__()

        channels.initialize_from_config(config)

        self.TX_position = config.getlist('setup', 'TX_coordinates')
        self.RX_position = config.getlist('setup', 'RX_coordinates')
        self.RIS_position = config.getlist('setup', 'RIS1_coordinates')
        self.N = int(np.prod(config.getlist('setup', 'RIS_elements')))
        self.dist_TX_RIS = np.linalg.norm(self.TX_position - self.RIS_position)
        self.dist_RIS_RX = np.linalg.norm(self.RIS_position - self.RX_position)

        self.theta_TX_RIS, self.phi_TX_RIS = ray_to_elevation_azimuth(self.TX_position, self.RIS_position)
        self.theta_RIS_RX, self.phi_RIS_RX = ray_to_elevation_azimuth(self.RIS_position, self.RX_position)

        self.episode_length = episode_length
        self.transmit_SNR = transmit_SNR
        self._t = None
        self._episode_ened = False
        self._prev_configuration = None

        action_dim = int(np.power(2, self.N))
        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int32,
                                                        minimum=0,
                                                        maximum=action_dim - 1,
                                                        name='action')

        self._observation_spec = array_spec.ArraySpec(shape=(self.N * 4 + self.N,),
                                                      dtype=np.float32,
                                                      name='observation')


    # def to_tf_py_env(self):
    #     return tf_py_environment.TFPyEnvironment(self)
    #
    #
    # def get_random_policy(self):
    #     #assert isinstance(self, TFPyEnvironment)
    #     return random_tf_policy.RandomTFPolicy(self.time_step_spec(), self.action_spec())


    @property
    def H(self):
        return channels.TX_RIS_channel_model(self.N, self.dist_TX_RIS, self.theta_TX_RIS, self.phi_TX_RIS,
                                             channels.element_spacing)

    @property
    def G(self):
        return channels.TX_RIS_channel_model(self.N, self.dist_RIS_RX, self.theta_RIS_RX, self.phi_RIS_RX,
                                             channels.element_spacing)

    def generate_observation_vector(self, H, G):
        H = H.reshape(1, self.N)
        G = G.reshape(1, self.N)
        # obs = np.vstack([H.real, H.imag, G.real, G.imag]).astype(np.float32).flatten()
        obs = np.vstack([H.real, H.imag, G.real, G.imag, self._prev_configuration]).astype(np.float32).flatten()
        return obs

    def compute_reward(self, H, G, Phi):
        return self.calculate_SNR(H, G, Phi, self.transmit_SNR)

    def action_2_configuration(self, a):
        bits = np.array([int(x) for x in list('{0:0b}'.format(a))], dtype=int)
        rem_padding = self.N - len(bits)
        zeros = np.zeros((rem_padding,), dtype=int)
        configuration = np.concatenate([zeros, bits])
        return configuration

    @staticmethod
    def configuration2phases(configuration):
        # thetas = configuration*np.pi # convert {0,1} to {0, pi}
        # return np.exp(1j*thetas)
        return 2 * configuration - 1  # exp(j*theta) for theta in {0, pi} results in {-1,1}. So better to do it manually for numerical stability

    @staticmethod
    def calculate_SNR(H, G, Phi, transmit_SNR=1):
        A = H * G
        B = np.dot(A, Phi)
        C = np.absolute(B)
        D = np.power(C, 2)
        snr = transmit_SNR * D
        return snr

    def seed(self, seed_=None):
        pass

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._t = 0
        self._episode_ended = False
        self._prev_configuration = np.zeros(shape=(self.N), dtype=np.float32)
        state = self.generate_observation_vector(self.H, self.G)
        return ts.restart(state)

    def _step(self, action):
        #         if action not in self.action_space:
        #             raise ValueError

        # print(f'Env: Got action {action}')

        if self._episode_ended:
            return self.reset()

        # configuration = action
        configuration = self.action_2_configuration(action)
        conf_str = ''.join(map(str, configuration))
        # print(f'Env: converted to configuration: {conf_str}')

        H_curr = self.H
        G_curr = self.G

        # print("Env: H: ",H_curr)

        Phi = self.configuration2phases(configuration)

        reward = self.compute_reward(H_curr, G_curr, Phi)
        # print(f'Env: reward: {reward}')

        self._prev_configuration = configuration

        observation = self.generate_observation_vector(H_curr, G_curr)

        self._t += 1

        # print('Env: t',self._t)

        if self.episode_length is not None and self._t >= self.episode_length:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward, discount=1.0)



class RISEnv2(py_environment.PyEnvironment):


    def __init__(self, setup: Setup, episode_length):
        super(RISEnv2, self).__init__()
        self.setup = setup

        self.episode_length      = episode_length
        self._t                  = None
        self._episode_ended      = False
        self._prev_configuration = None
        self.observation_noise_variance = setup.observation_noise_variance


        action_dim, observation_dim = self._calculate_space_dimensions()

        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int32,
                                                        minimum=0,
                                                        maximum=action_dim - 1,
                                                        name='action')

        self._observation_spec = array_spec.ArraySpec(shape=(observation_dim,),
                                                      dtype=np.float32,
                                                      name='observation')

    def _calculate_space_dimensions(self):

        M = self.setup.M
        S = self.setup.S
        K = self.setup.K
        N = self.setup.N
        #N_tot = self.setup.N_tot
        N_controllable = self.setup.N_controllable

        action_dim      = int(np.power(2, N_controllable))
        observation_dim = ((M * S * N) + (M * K * N) + (K * S)) *      2
        #                   +-------+    +---------+   +------+  +---------+
        #                       H             G           h       real+imag

        return action_dim, observation_dim


    def generate_observation_vector(self, H, G, h):
        H   = H.flatten()
        G   = G.flatten()
        h   = h.flatten()
        obs = np.concatenate([H.real, H.imag, G.real, G.imag, h.real, h.imag])
        obs = obs.astype(np.float32).flatten()

        if self.observation_noise_variance != 0:
            obs += np.random.normal(loc=0, scale=self.observation_noise_variance, size=len(obs))

        return obs

    def compute_reward(self, H, G, h, RIS_profiles):
        SINR = compute_SINR_per_user(self.setup,
                                     self.setup.noise_variance,
                                     H, G, h,
                                     RIS_profiles,
                                     self.setup.precoding_v)
        return float(np.sum(SINR))
        # return compute_sum_rate(SINR)

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

    def seed(self, seed_=None):
        pass

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._t = 0
        self._episode_ended = False
        self._prev_configuration = np.zeros(shape=self._observation_spec.shape, dtype=np.float32)
        H, G, h = simulate_transmission(self.setup)
        state = self.generate_observation_vector(H, G, h)
        return ts.restart(state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        H, G, h       = simulate_transmission(self.setup)
        configuration = self.action_2_configuration(action)
        configuration = self.extend_configuration_to_groups(configuration)
        Phi           = self.configuration2phases(configuration)
        reward        = self.compute_reward(H, G, h, Phi)
        observation   = self.generate_observation_vector(H, G, h)
        self._t      += 1

        if self.episode_length is not None and self._t >= self.episode_length:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward, discount=1.0)


    def get_info(self) -> Any:
        raise NotImplemented


# -------------------------------------------------------------------
class BanditWrapper:
    def __init__(self, tf_env):
        self.env = tf_env

    def generate_reward(self, action):
        time_step = self.env._step(action)
        return time_step.reward.numpy()[0]



