import json
import os

import matplotlib.pyplot as plt

from RL_experiments.experiments import Experiment, LinearMovementExperiment, get_experiment_class_from_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from dataclasses import dataclass
from typing import Callable, Tuple, List

from RL_experiments.training_utils import compute_baseline_scores, display_and_save_results, \
    AgentParams, Agent, apply_callbacks

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

    alpha             : int = None
    gamma             : int = None



class UCBAgent(Agent):


    def __init__(self, params: UCBParams, num_actions, observation_dim, observation_type="unused"):


        super().__init__("UCB", params, num_actions, observation_dim, "")
        self.name         = "UCB"
        self.params       = params
        self._num_actions = num_actions
        self._alpha       = self.params.alpha
        self._gamma       = self.params.gamma
        self.Q            = None # shape: (num_actions,)
        self.N            = None # shape: (num_actions,)
        self._t           = None

        #self.params.num_iterations = int(self.params.num_iterations * self._num_actions)
        self.restart()

    def restart(self):
        self.Q  = np.random.normal(loc=0, scale=0.001, size=(self._num_actions,))
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


    def _initialize_training_vars(self, env: RISEnv2):
        self.Q = 1 + np.random.normal(loc=0, scale=0.001, size=(self._num_actions,))
        self.N = np.zeros((self._num_actions,), dtype=np.int32)
        self._t = 1

    def _apply_collect_step(self, step, obs, action, reward):
        self.curr_action = action
        self.curr_reward = reward

    def _perform_update_step(self) ->List:
        self.update(self.curr_action, self.curr_reward)
        return []


    #
    #
    # def train(self, env: RISEnv2, callbacks=None):
    #     if callbacks is None: callbacks = []
    #
    #     eval_interval = self.params.num_iterations // self.params.num_evaluations
    #
    #     rewards = []
    #     reward_steps = []
    #     losses = []
    #
    #     initial_reward, _ = self.evaluate(env)
    #
    #     rewards.append(initial_reward)
    #     reward_steps.append(0)
    #
    #     time_step = env._reset()
    #     try:
    #         for step in tqdm(range(self.params.num_iterations)):
    #
    #             if time_step.is_last():
    #                 time_step = env._reset()
    #
    #             obs       = time_step.observation
    #             action    = self.collect_policy(obs)
    #             time_step = env._step(action)
    #             reward    = time_step.reward
    #
    #             self.update(action, reward)
    #
    #
    #             if (step + 1) % eval_interval == 0:
    #                 avg_score, std_score = self.evaluate(env)
    #                 tqdm.write(f"step={step} | Avg reward = {avg_score} +/- {std_score}.")
    #                 rewards.append(avg_score)
    #                 reward_steps.append(step)
    #
    #             converged_flag, converged_callback_names = apply_callbacks(callbacks, step, obs, action, reward)
    #             if converged_flag:
    #                 tqdm.write(f"Step={step} | Algorithm converged due to criteria: {converged_callback_names}")
    #                 break
    #
    #     except KeyboardInterrupt:
    #         print("Training stopped by user...")
    #
    #     return rewards, losses, reward_steps, self.policy
    #


if __name__ == '__main__':
    import sys
    params_filename = sys.argv[1]

    exp = get_experiment_class_from_config(params_filename)
    agent, info = exp.run(UCBAgent,
                   UCBParams,
                   "UCB_PARAMS", )

    import seaborn as sns
    plt.figure()
    plt.bar(range(len(agent.Q)), agent.Q)
    plt.xlabel("Action index")
    plt.ylabel("rate")
    plt.title("UCB reward predictions after training")
    plt.show(block=False)

    plt.figure()
    plt.bar(range(len(agent.Q)), agent.N)
    plt.xlabel("Action index")
    plt.title("Number of times each action was selected during training")
    plt.show(block=False)

    #
    # def plot_eval_statistics(info):
    #     rewards_per_action = {}
    #
    #     for a, r in zip(info['action'], info['reward']):
    #         if a not in rewards_per_action.keys():
    #             rewards_per_action[a] = (0, 0)
    #
    #         cum_reward, n_selected = rewards_per_action[a]
    #         rewards_per_action[a]  = (cum_reward + r, n_selected + 1)
    #
    #     selected_actions = []
    #     avg_rewards      = []
    #     for key, val in rewards_per_action.items():
    #         action = key
    #         selected_actions.append(action)
    #         cum_reward, n_selected = val
    #         avg_reward = cum_reward / n_selected
    #         avg_rewards.append(avg_reward)
    #
    #         print(f'action: {action} | n_selected: {n_selected} | avg_reward: {avg_reward}')
    #
    #     if len(selected_actions) > 1:
    #         plt.bar(selected_actions, avg_rewards)
    #         plt.show(block=False)
    #
    # plot_eval_statistics(info)