import json
import sys, os
from typing import List
import numpy as np

from RL_experiments.environments import RISEnv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from RL_experiments.training_utils import Agent, AgentParams


class RandomAgent(Agent):
    def __init__(self, params: AgentParams, num_actions, observation_dim="unused", observation_type='unused'):
        super(RandomAgent, self).__init__('RandomAgent', params, num_actions, observation_dim , "")

    @property
    def policy(self):
        def choose_random_action(obs):
            return np.random.choice(self.num_actions)
        return choose_random_action

    @property
    def collect_policy(self):
        return self.policy

    def _initialize_training_vars(self, env: RISEnv2):
        pass

    def _apply_collect_step(self, step, obs, action, reward) -> None:
        pass

    def _perform_update_step(self) -> List:
        return []



if __name__ == '__main__':
    from RL_experiments.experiments import Experiment

    import sys
    params_filename = sys.argv[1]

    exp = Experiment(params_filename)
    agent, info = exp.run_baseline(None, RandomAgent)