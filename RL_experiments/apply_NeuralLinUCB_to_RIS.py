import numpy as np
import matplotlib.pyplot as plt

from RL_experiments.test_environment import TestEnv
from utils.custom_configparser import CustomConfigParser
from RL_experiments.environments import RIS_TFenv, RISEnv2
from RL_experiments.training import NeuralLinUCBParams, initialize_Neural_Lin_UCB_agent, train_bandit_agent, \
    plot_training_performance, evaluate_agent, plot_loss
import RL_experiments.parameters as params
from RL_experiments.standalone_simulatiion import Setup
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy




setup1 = Setup(**params.SETUP)


#env           = RISEnv2(setup1, episode_length=10) #RIS_TFenv(config, 1, transmit_SNR=1)

env = TestEnv(actions=10)

train_env     = tf_py_environment.TFPyEnvironment(env)
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


num_eval_timesteps = 1000
random_policy_average_return = evaluate_agent(random_policy, train_env, num_eval_timesteps, name='Random')

nlucb_agent_name = 'Neural LinUCB'


nlucbParams = NeuralLinUCBParams(**params.NEURAL_LIN_UCB_PARAMS)
nlucb_agent = initialize_Neural_Lin_UCB_agent(nlucbParams, train_env)


reward_values_NLUCB, loss_infos = train_bandit_agent(nlucb_agent, train_env, nlucbParams)


plot_training_performance(reward_values_NLUCB, len(reward_values_NLUCB), 2*nlucbParams.log_interval, nlucb_agent_name, random_policy_average_return)
plot_loss(reward_values_NLUCB, nlucb_agent_name, scale='linear')


score = evaluate_agent(nlucb_agent.policy, train_env, 20, nlucb_agent_name)
score_as_percentage_of_random = (1 - score / random_policy_average_return) * 100
print(f'{nlucb_agent_name} attained mean performance of {score} ( {score_as_percentage_of_random}% improvement of random policy).')