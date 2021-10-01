import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from dataclasses import dataclass
from typing import Callable, Tuple

from RL_experiments.training_utils import compute_baseline_scores, display_and_save_results, evaluate_agent, \
    AgentParams, Agent, run_experiment

from tensorflow.keras import backend as K

import numpy as np
from tqdm import tqdm

from RL_experiments.environments import RISEnv2
from RL_experiments.standalone_simulatiion import Setup

import warnings
warnings.simplefilter('error')

@dataclass
class UCBParams(AgentParams):
    """
    `alpha`     (float) positive scalar. This is the exploration parameter that multiplies the confidence intervals.
    `gamma`     a float forgetting factor in [0.0, 1.0]. When set to 1.0, the algorithm does not forget.
    """

    alpha             : int
    gamma             : int



class UCBAgent(Agent):
    def __init__(self, params: UCBParams, num_actions, observation_dim):


        super().__init__("UCB", params, num_actions, observation_dim)
        self.name         = "UCB"
        self.params       = params
        self._num_actions = num_actions
        self._alpha       = self.params.alpha
        self._gamma       = self.params.gamma
        self.Q            = None # shape: (num_actions,)
        self.N            = None # shape: (num_actions,)
        self._t           = None

        self.params.num_iterations = int(self.params.num_iterations * self._num_actions)
        self.restart()

    def restart(self):
        self.Q  = np.random.normal(loc=0, scale=0.1, size=(self._num_actions,))
        self.N  = np.zeros((self._num_actions,), dtype=np.int32)
        self._t = 1

    def get_confidences(self):
        if np.all(self.Q == 0):
            return np.random.uniform(0, 1, size=self.Q.shape)
        else:
            confidences = np.zeros(self._num_actions)
            for a in range(self._num_actions):
                if self.N[a] != 0:
                    confidences[a] = np.sqrt(self._t) / self.N[a]
            return confidences

    def select_action(self):
        confidences = self.get_confidences()
        return np.argmax(self.Q + self._alpha * confidences)

    def select_action_greedy(self):
        return np.argmax(self.Q)

    def update(self, action, reward):
        if not (0 <= action < self._num_actions): raise ValueError

        self.N[action] += 1
        self._t += 1
        self.Q[action] = self.Q[action] + self._gamma *(reward - self.Q[action])


    @staticmethod
    def ignore_observation_policy_wrapper(_select_action_func: Callable)->Callable:
        def _policy(obs_):
            del obs_
            return _select_action_func()
        return _policy

    @property
    def policy(self) -> Callable:
        return self.ignore_observation_policy_wrapper(self.select_action_greedy)

    @property
    def collect_policy(self) -> Callable:
        return self.ignore_observation_policy_wrapper(self.select_action)


    def train(self, env: RISEnv2):

        eval_interval = self.params.num_iterations // self.params.num_evaluations

        rewards = []
        reward_steps = []
        losses = []

        initial_reward, _ = evaluate_agent(self, env)

        rewards.append(initial_reward)
        reward_steps.append(0)

        time_step = env._reset()
        try:
            for step in tqdm(range(self.params.num_iterations)):

                if time_step.is_last():
                    time_step = env._reset()

                obs       = time_step.observation
                action    = self.collect_policy(obs)
                time_step = env._step(action)
                reward    = time_step.reward

                self.update(action, reward)


                if (step + 1) % eval_interval == 0:
                    avg_score, std_score = evaluate_agent(self, env)
                    tqdm.write(f"step={step} | Avg reward = {avg_score} +/- {std_score}.")
                    rewards.append(avg_score)
                    reward_steps.append(step)

        except KeyboardInterrupt:
            print("Training stopped by user...")

        return rewards, losses, reward_steps, self.policy



if __name__ == '__main__':

    import sys
    run_experiment(sys.argv[1], UCBAgent, UCBParams, "UCB_PARAMS", "num_iterations,alpha")