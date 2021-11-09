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
class NeuralSoftmaxParams(AgentParams):
    fc_layer_params : Tuple     = None
    dropout_p       : float     = None
    steps_per_loop  : int       = None
    batch_size      : int       = None
    learning_rate   : float     = None
    Boltzmann_tau   : float     = None
    tau_change      : str       = None
    tau_initial     : float     = None
    tau_final       : float     = None

    def __post_init__(self):
        if self.tau_change is not None:
            if self.tau_change not in ['constant', 'linear', 'exponential']: raise ValueError






class NeuralSoftmaxAgent(Agent):

    def __init__(self, params: NeuralSoftmaxParams, num_actions: int, observation_dim: int):

        super().__init__("Neural Softmax", params, num_actions, observation_dim)
        self.batch_size = self.params.batch_size
        #self.params.num_iterations = int(self.params.num_iterations * num_actions)

        assert self.params.tau_change == 'constant'

        self.rewardNet = tf.keras.Sequential([
            tf.keras.layers.Input((observation_dim,)),
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


    def _argmax_action_selection(self, obs):
        obs = obs.reshape(1,-1)
        rewards = self.rewardNet.predict(obs)[0,:]
        return np.argmax(rewards)

    # def _epsilon_greedy_selection(self, obs):
    #     rnd = np.random.uniform(0, 1)
    #
    #     if rnd <= self.params.epsilon_greedy:
    #         return np.random.randint(0, self.num_actions)
    #     else:
    #         return self._argmax_action_selection(obs)

    def _Boltzmann_distribution_selection(self, obs):
        obs       = obs.reshape(1, -1)
        rewards   = self.rewardNet.predict(obs)[0, :]
        exponents = np.exp(rewards/self.params.Boltzmann_tau)
        probs     = exponents / np.sum(exponents)
        return np.random.choice(self.num_actions, p=probs)



    @property
    def policy(self)->Callable:
        return self._argmax_action_selection

    @property
    def collect_policy(self)->Callable:
        return self._Boltzmann_distribution_selection
    
    
    def _initialize_training_vars(self):
        self.batch_X   = np.empty((self.batch_size, self.observation_dim))
        self.batch_y   = np.empty((self.batch_size, 2))
        self.batch_i   = 0
        self.epoch_i   = 0
        
    def _apply_collect_step(self, step, obs, action, reward):

        self.batch_X[self.batch_i, :] = obs
        self.batch_y[self.batch_i, 0] = reward
        self.batch_y[self.batch_i, 1] = action
        self.batch_i                  = (self.batch_i + 1) % self.batch_size
        


    def _perform_update_step(self):
        if self.batch_i % self.batch_size == 0:
            hist = self.rewardNet.fit(self.batch_X, self.batch_y, batch_size=self.batch_size, epochs=self.params.steps_per_loop,
                                      steps_per_epoch=1, verbose=0)
            self.epoch_i += 1
            loss = hist.history['loss'][0]

            return [loss]
        else:
            return []







if __name__ == '__main__':
    import sys
    params_filename = sys.argv[1]


    exp = Experiment(params_filename)
    exp.run(NeuralSoftmaxAgent,
            NeuralSoftmaxParams,
            "NEURAL_SOFTMAX_PARAMS",)