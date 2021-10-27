import json
import os


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


from dataclasses import dataclass
from typing import Callable, Tuple

from RL_experiments.training_utils import compute_baseline_scores, display_and_save_results, \
    AgentParams, Agent, run_experiment, apply_callbacks

from tensorflow.keras import backend as K

import numpy as np
from tqdm import tqdm

from RL_experiments.environments import RISEnv2
from RL_experiments.standalone_simulatiion import Setup



@dataclass
class NeuralEpsilonGreedyParams(AgentParams):
    fc_layer_params : Tuple
    dropout_p       : float
    steps_per_loop  : int
    batch_size      : int
    learning_rate   : float
    epsilon_greedy  : float

    def __post_init__(self):
        pass






class CustomNeuralEpsilonGreedy(Agent):

    def __init__(self, params: NeuralEpsilonGreedyParams, num_actions: int, observation_dim: int):

        super().__init__("Neural ε-greedy", params, num_actions, observation_dim)
        self.batch_size = self.params.batch_size
        #self.params.num_iterations = int(self.params.num_iterations * num_actions)

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

    def _epsilon_greedy_selection(self, obs):
        rnd = np.random.uniform(0, 1)

        if rnd <= self.params.epsilon_greedy:
            return np.random.randint(0, self.num_actions)
        else:
            return self._argmax_action_selection(obs)


    @property
    def policy(self)->Callable:
        return self._argmax_action_selection

    @property
    def collect_policy(self)->Callable:
        return self._epsilon_greedy_selection

    def train(self, env: RISEnv2,callbacks=None):
        if callbacks is None: callbacks = []

        eval_interval = self.params.num_iterations // self.params.num_evaluations

        batch_X   = np.empty((self.batch_size, self.observation_dim))
        batch_y   = np.empty((self.batch_size, 2))
        time_step = env._reset()
        batch_i   = 0
        epoch_i   = 0

        rewards      = []
        reward_steps = []
        losses       = []

        initial_reward, _ = self.evaluate(env)

        rewards.append(initial_reward)
        reward_steps.append(0)



        try:
            for step in tqdm(range(self.params.num_iterations)):

                if time_step.is_last():
                    time_step = env._reset()

                obs       = time_step.observation
                action    = self.collect_policy(obs)
                time_step = env._step(action)
                reward    = time_step.reward

                batch_X[batch_i, :] = obs
                batch_y[batch_i, 0] = reward
                batch_y[batch_i, 1] = action
                batch_i             = (batch_i + 1) % self.batch_size

                if batch_i % self.batch_size == 0:
                    hist = self.rewardNet.fit(batch_X, batch_y, batch_size=self.batch_size, epochs=self.params.steps_per_loop, steps_per_epoch=1, verbose=0)
                    epoch_i += 1
                    losses.append(hist.history['loss'][0])

                if (step) % eval_interval == 0:
                    avg_score, std_score = self.evaluate(env)
                    tqdm.write(f"step={step} | Avg reward = {avg_score} +/- {std_score}.")
                    rewards.append(avg_score)
                    reward_steps.append(step)

                converged_flag, converged_callback_names = apply_callbacks(callbacks, step, obs, action, reward)
                if converged_flag:
                    tqdm.write(f"Step={step} | Algorithm converged due to criteria: {converged_callback_names}")
                    break

        except KeyboardInterrupt:
            print("Training stopped by user...")


        return rewards, losses, reward_steps, self.policy






if __name__ == '__main__':
    import sys
    run_experiment(sys.argv[1],
                   CustomNeuralEpsilonGreedy,
                   NeuralEpsilonGreedyParams,
                   "NEURAL_EPSILON_GREEDY_PARAMS",
                   "num_iterations,learning_rate")