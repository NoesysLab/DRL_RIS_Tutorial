import json
import os

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from dataclasses import dataclass
from typing import Callable, Tuple

from RL_experiments.training_utils import compute_baseline_scores, display_and_save_results, \
    AgentParams, Agent, apply_callbacks

from tensorflow.keras import backend as K

import numpy as np
from tqdm import tqdm

from RL_experiments.environments import RISEnv2
from RL_experiments.standalone_simulatiion import Setup
from RL_experiments.experiments import Experiment


@dataclass
class RNNPredictionBanditParams(AgentParams):
    fc_layer_params   : Tuple = None
    dropout_p         : float = None
    steps_per_loop    : int = None
    batch_size        : int = None
    learning_rate     : float = None
    Boltzmann_tau     : float = None
    tau_change        : str = None
    tau_initial       : float = None
    tau_final         : float = None
    lookback          : int = None

    def __post_init__(self):
        if self.tau_change is not None:
            if self.tau_change not in ['constant', 'linear', 'exponential']: raise ValueError


class RNNPredictionBandit(Agent):

    def __init__(self, params: RNNPredictionBanditParams, num_actions: int, observation_dim: int):

        super().__init__("RNN Predictor", params, num_actions, observation_dim)
        self.batch_size = self.params.batch_size

        # for type hinting only
        self.params = params # type: RNNPredictionBanditParams


        # self.params.num_iterations = int(self.params.num_iterations * num_actions)

        assert self.params.tau_change == 'constant'

        self.rewardNet = tf.keras.Sequential([
            tf.keras.layers.Input((self.params.lookback, observation_dim)),
            tf.keras.layers.Reshape((-1, 1, 1)),
            tf.keras.layers.Conv2D(128, (5, 1), data_format="channels_last"),
            tf.keras.layers.Dropout(self.params.dropout_p),
            tf.keras.layers.MaxPool2D((16, 1)),
            tf.keras.layers.Conv2D(128, (5, 1), data_format="channels_last"),
            tf.keras.layers.Dropout(self.params.dropout_p),
            tf.keras.layers.MaxPool2D((16, 1)),
            tf.keras.layers.Conv2D(32, (5, 1), data_format="channels_last"),
            tf.keras.layers.Dropout(self.params.dropout_p),
            tf.keras.layers.MaxPool2D((4, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(self.params.dropout_p),
            tf.keras.layers.Dense(1034, activation='relu'),
            tf.keras.layers.Dropout(self.params.dropout_p),
            tf.keras.layers.Dense(num_actions, activation='linear')
        ])

        def custom_MSE(y_actual, y_pred):
            i = tf.cast(y_actual[:, 1], tf.int32)
            r = y_actual[:, 0]
            r_pred = tf.gather(y_pred, i, axis=1)

            # print(r.shape)
            # print(y_pred.shape)
            # print(r_pred.shape)
            # print(i.shape)

            return K.mean((r - r_pred) ** 2)

        optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
        self.rewardNet.compile(optimizer, loss=custom_MSE)


    def apply_reward_prediction(self, obs, past_observations=None):
        if past_observations is None:
            past_observations = self.lookback_observations.copy()
        past_observations[:self.params.lookback - 1, :] = past_observations[1:, :]
        past_observations[-1, :] = obs
        past_observations = past_observations[np.newaxis, : , :]
        rewards = self.rewardNet.predict(past_observations)[0, :]
        return rewards

    def _argmax_action_selection(self, obs):
        predicted_rewards = self.apply_reward_prediction(obs)
        return np.argmax(predicted_rewards)


    def _Boltzmann_distribution_selection(self, obs):
        redicted_rewards = self.apply_reward_prediction(obs)
        exponents = np.exp(redicted_rewards / self.params.Boltzmann_tau)
        probs = exponents / np.sum(exponents)
        return np.random.choice(self.num_actions, p=probs)

    @property
    def policy(self) -> Callable:
        return self._argmax_action_selection

    @property
    def collect_policy(self) -> Callable:
        return self._Boltzmann_distribution_selection

    def _initialize_training_vars(self, env: RISEnv2):
        self.lookback_observations = np.zeros((self.params.lookback, self.observation_dim))
        self.batch_X = np.empty((self.batch_size, self.params.lookback, self.observation_dim))
        self.batch_y = np.empty((self.batch_size, 2))
        self.batch_i = 0
        self.epoch_i = 0

    def _apply_collect_step(self, step, obs, action, reward):
        self.lookback_observations[:self.params.lookback-1,:] = self.lookback_observations[1:,:]
        self.lookback_observations[-1,:]                      = obs

        self.batch_X[self.batch_i, :,:] = self.lookback_observations
        self.batch_y[self.batch_i, 0] = reward
        self.batch_y[self.batch_i, 1] = action
        self.batch_i = (self.batch_i + 1) % self.batch_size

    def _perform_update_step(self):

        if self.epoch_i < self.params.lookback:
            return []


        if self.batch_i % self.batch_size == 0:
            hist = self.rewardNet.fit(self.batch_X, self.batch_y, batch_size=self.batch_size,
                                      epochs=self.params.steps_per_loop,
                                      steps_per_epoch=1, verbose=0)
            self.epoch_i += 1
            loss = hist.history['loss'][0]

            return [loss]
        else:
            return []

    def evaluate(self, env: RISEnv2, return_info=False):
        rewards = np.empty((self.params.num_eval_episodes,))
        time_step = env._reset()

        info = {'observation': [], 'action': [], 'reward': []}

        for i in range(self.params.num_eval_episodes):
            if time_step.is_last(): time_step = env._reset()

            obs        = time_step.observation
            action     = self.policy(obs)
            time_step  = env._step(action)
            reward     = time_step.reward
            rewards[i] = reward

            self._apply_collect_step(None, obs, action, reward)

            if return_info:
                info['observation'].append(obs)
                info['action'].append(action)
                info['reward'].append(reward)

        if return_info:
            return rewards.mean(), rewards.std(), info
        else:
            return rewards.mean(), rewards.std()


if __name__ == '__main__':
    import sys

    params_filename = sys.argv[1]

    exp = Experiment(params_filename)
    exp.run(RNNPredictionBandit,
            RNNPredictionBanditParams,
            "RNN_PREDICTION_PARAMS", )