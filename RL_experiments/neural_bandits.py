import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import seaborn as sns
from typing import *
from scipy.constants import pi, c as speed_of_light
import configparser
from tqdm import tqdm

from tf_agents.agents import tf_agent
from tf_agents.drivers import driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.policies import random_tf_policy
from tf_agents import networks



from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts




from tf_agents.bandits.agents import lin_ucb_agent, neural_linucb_agent, neural_epsilon_greedy_agent
from tf_agents.bandits.environments import stationary_stochastic_py_environment as sspe
from tf_agents.bandits.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import matplotlib.pyplot as plt

from utils.custom_configparser import CustomConfigParser
from core import channels
from utils.misc import ray_to_elevation_azimuth

config = CustomConfigParser(interpolation=configparser.ExtendedInterpolation(),
                            allow_no_value=True,
                            inline_comment_prefixes=('#',))
fin = open('./setup_config.ini', 'r')
config.read_file(fin)
config.print()


channels.initialize_from_config(config)


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


def train_bandit_agent(agent,
                       environment,
                       num_iterations,
                       steps_per_loop=1,
                       batch_size=1,
                       log_interval=20):
    class RewardObserver:
        def __init__(self):
            self.reward_values = np.zeros(num_iterations)
            self.it_cnt = 0

        def __call__(self, trajectory):
            curr_reward = float(trajectory.reward.numpy())
            self.reward_values[self.it_cnt] = curr_reward

            self.it_cnt += 1
            if self.it_cnt % log_interval == 0:
                last_window_rewards = self.reward_values[self.it_cnt - log_interval:self.it_cnt]
                avg_reward = last_window_rewards.mean()
                tqdm.write('Iteration {}: Window avg reward: {}'.format(self.it_cnt,
                                                                   avg_reward))

    reward_observer = RewardObserver()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.policy.trajectory_spec,
        batch_size=batch_size,
        max_length=steps_per_loop)

    observers = [replay_buffer.add_batch, reward_observer]

    driver = dynamic_step_driver.DynamicStepDriver(
        env=environment,
        policy=agent.collect_policy,
        num_steps=steps_per_loop * batch_size,
        observers=observers)

    try:
        for _ in tqdm(range(num_iterations)):
            driver.run()
            loss_info = agent.train(replay_buffer.gather_all())
            replay_buffer.clear()

    except KeyboardInterrupt:
        print('Training stopped by user...')

    return reward_observer.reward_values, reward_observer.it_cnt




def plot_training_performance(reward_values, it_cnt, rolling_window, name=None, random_avg_reward=None):

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    name = name if name is not None else 'Trained agent'

    ma_rewards = moving_average(reward_values[:it_cnt], rolling_window)
    plt.figure(figsize=(30,12))
    #plt.plot(range(it_cnt), reward_values[:it_cnt], alpha=.7)
    plt.plot(range(rolling_window-1, it_cnt), ma_rewards, alpha=.7)
    if random_avg_reward is not None:
        plt.hlines([random_avg_reward], 0, it_cnt, color='k', ls=':', lw=5)
        plt.legend([name, 'Random policy'], fontsize=30)
    else:
        plt.legend([name], fontsize=30)

    plt.ylabel('Reward')
    plt.xlabel('Number of Iterations')

    plt.show()


def evaluate_agent(agent_policy, environment, num_timesteps, name=None):
    trained_agent_rewards = np.zeros(num_timesteps)
    time_step = environment.reset()
    for i in range(num_timesteps):
        action = agent_policy.action(time_step)
        time_step = environment.step(action)
        trained_agent_rewards[i] = float(time_step.reward)

    name = name if name is not None else 'Trained agent'

    print('\n\n{} policy average reward: {:2e} ± {:2e}\n'.format(
        name,
        trained_agent_rewards.mean(),
        trained_agent_rewards.std()
    ))

    return trained_agent_rewards.mean()




#######################################################################################
#######################################################################################
######################################################################################



# --------------------------------------------------------------
env = RIS_TFenv(config, None, transmit_SNR=1)#10e18)
environment = tf_py_environment.TFPyEnvironment(env)

random_policy = random_tf_policy.RandomTFPolicy(environment.time_step_spec(),
                                                environment.action_spec())
random_avg_reward = evaluate_agent(random_policy, environment, 1000, name='Random')

# ------------------------------------------------------------------------------------








# -------------------------------------------------------------------------------------

# agent_name = 'Linear UCB'
#
#
# env = RIS_TFenv(config, None, transmit_SNR=1)#10e18)
# environment = tf_py_environment.TFPyEnvironment(env)
#
#
# num_iterations = 100*int(2**env.N) # @param
# steps_per_loop = 1  # @param
# batch_size     = 1 # @param
# log_interval   = 20
#
#
# observation_spec = environment.observation_spec()
# time_step_spec = ts.time_step_spec(observation_spec)
# action_spec = environment.action_spec()
#
# agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=time_step_spec,
#                                      action_spec=action_spec)
#
#
# reward_values, it_cnt = train_bandit_agent(agent, environment, num_iterations, steps_per_loop, batch_size, log_interval)
#
# plot_training_performance(reward_values, it_cnt, 2*log_interval, agent_name, random_avg_reward)
# evaluate_agent(agent.policy, environment, 1000, agent_name)


# ----------------------------------------------------------------------------------------------------










# ----------------------------------------------------------------------------------------------

nlucb_agent_name = 'Neural LinUCB'

env = RIS_TFenv(config, None, transmit_SNR=1)#10e18)
environment = tf_py_environment.TFPyEnvironment(env)

num_actions = int(2 ** env.N)
num_iterations = 2000#100 * num_actions  # @param
steps_per_loop = 1  # @param
batch_size = 1  # @param
log_interval = 20

encoding_dim = 64
learning_rate = 10e-3

observation_spec = environment.observation_spec()
time_step_spec = ts.time_step_spec(observation_spec)
action_spec = environment.action_spec()

observation_dim = environment.observation_spec().shape.as_list()[0]


encoding_network = networks.encoding_network.EncodingNetwork(
    input_tensor_spec = observation_spec,
    fc_layer_params   = (100,100,encoding_dim),
)


nlucb_agent = neural_linucb_agent.NeuralLinUCBAgent(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    encoding_network=encoding_network,
    encoding_network_num_train_steps=2000,
    encoding_dim=encoding_dim,
    epsilon_greedy=0.1,
    alpha=5,
    gamma=0.5,
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

)

reward_values_NLUCB, it_cnt_NLUCB = train_bandit_agent(nlucb_agent, environment, num_iterations, steps_per_loop, batch_size, log_interval)

plot_training_performance(reward_values_NLUCB, it_cnt_NLUCB, 1, nlucb_agent_name, random_avg_reward)
evaluate_agent(nlucb_agent.policy, environment, 10, nlucb_agent_name)

# ----------------------------------------------------------------------------------------------------------




# # # # # # # # # # # # # # # #
# CHECKPOINT
# # # # # # # # # # # # # # # # #