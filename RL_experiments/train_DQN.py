import json
import os
from copy import deepcopy
from typing import Tuple, Callable, Union
import numpy as np
import matplotlib.pyplot as plt
import builtins
from dataclasses import dataclass, field
import tensorflow as tf
import tf_agents
from tf_agents import networks, bandits
from tf_agents.agents import TFAgent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks import sequential
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils.common import element_wise_squared_loss, element_wise_huber_loss
from tf_agents.trajectories import time_step as ts
from tf_agents.bandits.agents.neural_epsilon_greedy_agent import NeuralEpsilonGreedyAgent
from tf_agents.networks import network
from scipy.interpolate import make_interp_spline, BSpline
from tqdm import tqdm

from RL_experiments.training_utils import compute_avg_return


@dataclass()
class DQNParams:
    fc_layer_params            : Tuple
    num_iterations             : int
    initial_collect_steps      : int
    collect_steps_per_iteration: int
    replay_buffer_max_length   : int
    batch_size                 : int
    learning_rate              : float
    log_interval               : int
    eval_interval              : int
    epsilon_greedy             : float
    gradient_clipping          : float
    n_step_update              : int
    target_update_tau          : float
    target_update_period       : int
    gamma                      : float
    num_eval_episodes          : int
    num_actions                : int
    td_errors_loss_fn          : Callable = element_wise_squared_loss

    def __post_init__(self):
        #self.num_iterations *= self.num_actions
        pass

def _construct_Q_network(num_actions, fc_layer_params):
    def dense_layer(num_units):
          return tf.keras.layers.Dense(
              num_units,
              activation           = tf.keras.activations.relu,
              #kernel_regularizer   = tf.keras.regularizers.l1(10e-3),
              #activity_regularizer = tf.keras.regularizers.l2(.2),
              kernel_initializer   = tf.keras.initializers.VarianceScaling(scale=5.0, mode='fan_in',
                                                                           distribution='truncated_normal'),
          )


    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation         = None,
        kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer   = tf.keras.initializers.Constant(-0.2),
        )

    return sequential.Sequential(dense_layers + [q_values_layer])


def initialize_DQN_agent(params: DQNParams, train_env: TFPyEnvironment):
    num_actions = int(train_env.action_spec().maximum) - int(train_env.action_spec().minimum) + 1
    params.num_actions = num_actions
    params.num_iterations = int(params.num_iterations *  params.num_actions)


    q_net        = _construct_Q_network(num_actions, params.fc_layer_params)
    target_q_net = _construct_Q_network(num_actions, params.fc_layer_params)
    optimizer    = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        target_q_network=target_q_net,
        optimizer=optimizer,
        train_step_counter=train_step_counter,
        td_errors_loss_fn=params.td_errors_loss_fn,
        epsilon_greedy=params.epsilon_greedy,
        gradient_clipping=params.gradient_clipping,
        n_step_update=params.n_step_update,
        target_update_tau=params.target_update_tau,
        target_update_period=params.target_update_period,
        gamma=params.gamma,

    )

    agent.initialize()

    return agent




def _collect_step(environment, policy, buffer):
    time_step      = environment.current_time_step()
    action_step    = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj           = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

def _collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        _collect_step(env, policy, buffer)

def train_DQN_agent(agent: dqn_agent.DqnAgent,
                    train_env: TFPyEnvironment,
                    params: DQNParams,
                    random_policy: TFPolicy):

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=params.replay_buffer_max_length)

    _collect_data(train_env, random_policy, replay_buffer, params.initial_collect_steps)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=params.batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = tf_agents.utils.common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return, _ = compute_avg_return(train_env, agent.policy, params.num_eval_episodes)
    returns          = [avg_return]
    train_losses     = []
    eval_steps       = [0]

    print('Starting training')

    try:
        for iter_cnt in tqdm(range(params.num_iterations)):
            _collect_data(train_env, agent.collect_policy, replay_buffer, params.collect_steps_per_iteration)
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss
            step = agent.train_step_counter.numpy()

            train_losses.append(train_loss)

            if step % params.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % params.eval_interval == 0:
                avg_return, std_return = compute_avg_return(train_env, agent.policy, params.num_eval_episodes)
                tqdm.write('step = {0}: Average Return = {1:.4f} +/- {2:.3f}'.format(step, avg_return, std_return))
                returns.append(avg_return)

    except KeyboardInterrupt:
        print('Training aborted by user...')
        num_iterations = iter_cnt


    return returns, train_losses, eval_steps