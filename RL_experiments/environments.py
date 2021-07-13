import sys
sys.path.insert(0,'..')

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts


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

    @property
    def H(self):
        return channels.TX_RIS_channel_model(self.N, self.dist_TX_RIS, self.theta_TX_RIS, self.phi_TX_RIS,
                                             channels.element_spacing)

    @property
    def G(self):
        return channels.TX_RIS_channel_model(self.N, self.dist_RIS_RX, self.theta_RIS_RX, self.phi_RIS_RX,
                                             channels.element_spacing)

    def generate_obsevation_vector(self, H, G):
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
        state = self.generate_obsevation_vector(self.H, self.G)
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

        observation = self.generate_obsevation_vector(H_curr, G_curr)

        self._t += 1

        # print('Env: t',self._t)

        if self.episode_length is not None and self._t >= self.episode_length:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward, discount=1.0)



class BanditWrapper:
    def __init__(self, tf_env):
        self.env = tf_env

    def generate_reward(self, action):
        time_step = self.env._step(action)
        return time_step.reward.numpy()[0]


# from gym import spaces
# class RIS_env(gym.Env):
#     def __init__(self, config: CustomConfigParser, episode_length=None, transmit_SNR=1):
#         super(RIS_env, self).__init__()
#
#         channels.initialize_from_config(config)
#
#         self.TX_position = config.getlist('setup', 'TX_coordinates')
#         self.RX_position = config.getlist('setup', 'RX_coordinates')
#         self.RIS_position = config.getlist('setup', 'RIS1_coordinates')
#         self.N = int(np.prod(config.getlist('setup', 'RIS_elements')))
#         self.dist_TX_RIS = np.linalg.norm(self.TX_position - self.RIS_position)
#         self.dist_RIS_RX = np.linalg.norm(self.RIS_position - self.RX_position)
#
#         self.theta_TX_RIS, self.phi_TX_RIS = ray_to_elevation_azimuth(self.TX_position, self.RIS_position)
#         self.theta_RIS_RX, self.phi_RIS_RX = ray_to_elevation_azimuth(self.RIS_position, self.RX_position)
#
#         self.episode_length = episode_length
#         self.transmit_SNR = transmit_SNR
#         self.timestep = None
#
#         # self.action_space      = spaces.MultiBinary(self.N)
#         action_dim = int(np.power(2, self.N))
#         if action_dim <= 0:
#             raise ValueError("Too high number of elements - arithmetic overflow in 2^N action space dimension.")
#         self.action_space = spaces.Discrete(action_dim)
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, self.N))
#
#     @property
#     def H(self):
#         return channels.TX_RIS_channel_model(N, dist_TX_RIS, theta_TX_RIS, phi_TX_RIS, channels.element_spacing)
#
#     @property
#     def G(self):
#         return channels.TX_RIS_channel_model(N, dist_RIS_RX, theta_RIS_RX, phi_RIS_RX, channels.element_spacing)
#
#     def channel_states_2_obsevation(self, H, G):
#         H = H.reshape(1, self.N)
#         G = G.reshape(1, self.N)
#         obs = np.vstack([H.real, H.imag, G.real, G.imag])
#         return obs
#
#     def compute_reward(self, H, G, Phi):
#         return calculate_SNR(H, G, Phi, self.transmit_SNR)
#
#     def action_2_configuration(self, a):
#         bits = np.array([int(x) for x in list('{0:0b}'.format(a))], dtype=int)
#         rem_padding = self.N - len(bits)
#         zeros = np.zeros((rem_padding,), dtype=int)
#         configuration = np.concatenate([zeros, bits])
#         return configuration
#
#     def seed(self, seed_=None):
#         pass
#
#     def reset(self):
#         self.timestep = 0
#         return self.channel_states_2_obsevation(self.H, self.G)
#
#     def step(self, action):
#         if action not in self.action_space:
#             raise ValueError
#
#         # configuration = action
#         configuration = self.action_2_configuration(action)
#
#         H_curr = self.H
#         G_curr = self.G
#
#         Phi = configuration2phases(configuration)
#
#         reward = self.compute_reward(H_curr, G_curr, Phi)
#
#         observation = self.channel_states_2_obsevation(H_curr, G_curr)
#
#         self.timestep += 1
#         if self.episode_length is not None and self.timestep >= self.episode_length:
#             done = True
#         else:
#             done = False
#
#         return observation, reward, done, {}