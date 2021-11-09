import numpy as np
import matplotlib.pyplot as plt

from utils.custom_configparser import CustomConfigParser
from RL_experiments.environments import RIS_TFenv, RISEnv2
from RL_experiments.training import train_bandit_agent, plot_training_performance, evaluate_agent

from RL_experiments.standalone_simulatiion import Setup

from tf_agents import utils
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import time_step as ts


from tf_agents.bandits.agents import lin_ucb_agent, neural_linucb_agent, neural_epsilon_greedy_agent


# config = CustomConfigParser().load_from_file(open('setup_config.ini', 'r'))
# config.print()

setup1        = Setup()
env           = RISEnv2(setup1, episode_length=10) #RIS_TFenv(config, 1, transmit_SNR=1)
train_env     = tf_py_environment.TFPyEnvironment(env)
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


num_eval_timesteps = 1000
random_policy_average_return = evaluate_agent(random_policy,
                                              train_env,
                                              num_eval_timesteps,
                                              name='Random')

num_actions = int(train_env.action_spec().maximum)-int(train_env.action_spec().minimum)+1


lucb_agent_name = 'Linear UCB'

num_iterations = 100*num_actions
steps_per_loop = 1
batch_size     = 1
log_interval   = 20


lucb_agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=ts.time_step_spec(train_env.observation_spec()),
                                          action_spec=train_env.action_spec(),
                                          alpha=1,)


reward_values_LUCB, it_cnt_LUCB = train_bandit_agent(lucb_agent, train_env, num_iterations, steps_per_loop, batch_size, log_interval)

plot_training_performance(reward_values_LUCB, it_cnt_LUCB, 2*log_interval, lucb_agent_name, random_policy_average_return)
score = evaluate_agent(lucb_agent.policy, train_env, 20, lucb_agent_name)

print(f'{lucb_agent_name} attained mean performance of {score}.')