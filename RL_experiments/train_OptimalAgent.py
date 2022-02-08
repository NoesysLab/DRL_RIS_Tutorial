import json
import sys, os
from typing import List
import numpy as np
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from RL_experiments.environments import RISEnv2, find_best_action_exhaustively




from RL_experiments.training_utils import Agent, AgentParams


class OptimalAgent(Agent):
    def __init__(self, params: AgentParams, num_actions, observation_dim="unused", observation_type='unused'):
        super(OptimalAgent, self).__init__('OptimalAgent', params, num_actions, observation_dim, "")
        self.env = None # type: RISEnv2




    def _initialize_training_vars(self, env: RISEnv2):
        self.env = env

    @property
    def policy(self):
        def exhaustive_search_single_realization(obs):
            r_max, a_best = find_best_action_exhaustively(self.env)
            return a_best
        return exhaustive_search_single_realization

    @property
    def collect_policy(self):
        def random_selection(obs):
            return np.random.choice(self.num_actions)
        return random_selection

    def _apply_collect_step(self, step, obs, action, reward) ->None:
        pass

    def _perform_update_step(self) ->List:
        return []

    def evaluate(self, env: RISEnv2, return_info=False, n_iters=None, verbose=False):
        original_env = self.env
        self.env = env
        rewards_mean, rewards_std = super().evaluate(env, False, n_iters, verbose)
        self.env = original_env
        return rewards_mean, rewards_std


if __name__ == '__main__':
    from RL_experiments.experiments import Experiment
    import sys
    params_filename = sys.argv[1]

    exp = Experiment(params_filename)
    agent, info = exp.run_baseline(None, OptimalAgent )