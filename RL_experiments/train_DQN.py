import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


from typing import Tuple, Callable, Union
from dataclasses import dataclass
from typing import Tuple, Callable

import tensorflow as tf
import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils.common import element_wise_squared_loss, element_wise_huber_loss
from tqdm import tqdm

from RL_experiments.OLD_VERSION.training import compute_avg_return
from RL_experiments.environments import RISEnv2
from RL_experiments.training_utils import AgentParams, Agent, run_experiment


@dataclass
class DQNParams(AgentParams):
    fc_layer_params            : Tuple
    initial_collect_steps      : int
    collect_steps_per_iteration: int
    replay_buffer_max_length   : int
    batch_size                 : int
    learning_rate              : float
    log_interval               : int
    epsilon_greedy             : float
    gradient_clipping          : float
    n_step_update              : int
    target_update_tau          : float
    target_update_period       : int
    gamma                      : float
    dropout_p                  : float
    td_errors_loss_fn          : str

    def __post_init__(self):
        #self.num_iterations *= self.num_actions
        pass



def map_name2loss(loss_function_name: str)->Callable:
    if loss_function_name.lower() in {'l2', 'mse', 'mean_squared_error', 'element_wise_squared_loss'}:
        return element_wise_squared_loss
    elif loss_function_name.lower() in {'huber', 'element_wise_huber_loss'}:
        return element_wise_huber_loss




class DQNAgent(Agent):
    def __init__(self, params: DQNParams, num_actions: int, observation_dim: int):
        super().__init__("DQN", params, num_actions, observation_dim)

        self.batch_size            = self.params.batch_size
        self.params.num_iterations = int(self.params.num_iterations * num_actions)


        self._tf_agent = None


    @property
    def policy(self) -> Callable:
        return self._tf_agent.policy

    @property
    def collect_policy(self) -> Callable:
        return self._tf_agent.collect_policy

    def _construct_Q_network(self, num_actions, fc_layer_params):
        # def dense_layer(num_units):
        #       return tf.keras.layers.Dense(
        #           num_units,
        #           activation           = tf.keras.activations.relu,
        #           #kernel_regularizer   = tf.keras.regularizers.l1(10e-3),
        #           #activity_regularizer = tf.keras.regularizers.l2(.2),
        #           kernel_initializer   = tf.keras.initializers.VarianceScaling(scale=5.0, mode='fan_in',
        #                                                                        distribution='truncated_normal'),
        #       )
        #
        #
        # dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
        # q_values_layer = tf.keras.layers.Dense(
        #     num_actions,
        #     activation         = None,
        #     kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        #     bias_initializer   = tf.keras.initializers.Constant(-0.2),
        #     )
        #
        # return sequential.Sequential(dense_layers + [q_values_layer])

        q_net_layers = [
            # tf.keras.layers.Input((observation_dim,)),
            tf.keras.layers.Reshape((-1, 1, 1)),
            tf.keras.layers.Conv2D(64, (5, 1), data_format="channels_last"),
            tf.keras.layers.Dropout(self.params.dropout_p),
            tf.keras.layers.MaxPool2D((4, 1)),
            tf.keras.layers.Conv2D(64, (5, 1), data_format="channels_last"),
            tf.keras.layers.Dropout(self.params.dropout_p),
            tf.keras.layers.MaxPool2D((4, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(self.params.dropout_p),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(self.params.dropout_p),
            tf.keras.layers.Dense(num_actions, activation='linear')
        ]

        return sequential.Sequential(q_net_layers)

    def initialize_DQN_agent(self, train_env: TFPyEnvironment):
        num_actions = int(train_env.action_spec().maximum) - int(train_env.action_spec().minimum) + 1
        self.params.num_actions = num_actions

        q_net = self._construct_Q_network(num_actions, self.params.fc_layer_params)
        target_q_net = self._construct_Q_network(num_actions, self.params.fc_layer_params)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)

        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DqnAgent(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            q_network=q_net,
            target_q_network=target_q_net,
            optimizer=optimizer,
            train_step_counter=train_step_counter,
            td_errors_loss_fn=map_name2loss(self.params.td_errors_loss_fn),
            epsilon_greedy=self.params.epsilon_greedy,
            gradient_clipping=self.params.gradient_clipping,
            n_step_update=self.params.n_step_update,
            target_update_tau=self.params.target_update_tau,
            target_update_period=self.params.target_update_period,
            gamma=self.params.gamma,

        )

        agent.initialize()

        return agent





    @staticmethod
    def _collect_data(env, policy, buffer, steps):
        for _ in range(steps):
            time_step = env.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            buffer.add_batch(traj)


    def train(self, env: RISEnv2):

        eval_interval = self.params.num_iterations // self.params.num_evaluations

        train_env = tf_py_environment.TFPyEnvironment(env)
        random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

        self._tf_agent = self.initialize_DQN_agent(train_env)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self._tf_agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=self.params.replay_buffer_max_length)

        self._collect_data(train_env, random_policy, replay_buffer, self.params.initial_collect_steps)

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.params.batch_size,
            num_steps=2).prefetch(3)

        iterator = iter(dataset)

        self._tf_agent.train = tf_agents.utils.common.function(self._tf_agent.train)
        self._tf_agent.train_step_counter.assign(0)

        rewards      = []
        reward_steps = []
        losses       = []

        initial_reward, _ = self.evaluate(train_env)

        rewards.append(initial_reward)
        reward_steps.append(0)



        print('Starting training')

        try:
            for _ in tqdm(range(self.params.num_iterations)):

                self._collect_data(train_env, self._tf_agent.collect_policy, replay_buffer, self.params.collect_steps_per_iteration)

                experience, _ = next(iterator)
                train_loss    = self._tf_agent.train(experience).loss
                step          = self._tf_agent.train_step_counter.numpy()

                losses.append(train_loss)

                if step % self.params.log_interval == 0:
                    print('step = {0}: loss = {1}'.format(step, train_loss))

                if (step+1) % eval_interval == 0:
                    avg_score, std_score = self.evaluate(train_env)
                    tqdm.write('step = {0}: Average Return = {1:.4f} +/- {2:.3f}'.format(step-1, avg_score, std_score))
                    rewards.append(avg_score)
                    reward_steps.append(step)

        except KeyboardInterrupt:
            print('Training aborted by user...')


        return rewards, losses, reward_steps, self.policy



    def evaluate(self, env: Union[RISEnv2, TFPyEnvironment]):
        if not isinstance(env, TFPyEnvironment):
            env = tf_py_environment.TFPyEnvironment(env)
        return compute_avg_return(env, self.policy, self.params.num_eval_episodes)




if __name__ == '__main__':
    import sys

    run_experiment(sys.argv[1],
                   DQNAgent,
                   DQNParams,
                   "DQN_PARAMS",
                   "num_iterations,learning_rate")