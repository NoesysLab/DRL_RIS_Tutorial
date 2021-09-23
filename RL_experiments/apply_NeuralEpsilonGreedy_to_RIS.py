import numpy as np
import matplotlib.pyplot as plt

from RL_experiments.test_environment import TestEnv
from utils.custom_configparser import CustomConfigParser
from RL_experiments.environments import RIS_TFenv, RISEnv2
from RL_experiments.training import NeuralEpsilonGreedyAgent, NeuralEpsilonGreedyParams, initialize_NeuralEpsilonGreedyAgent, train_bandit_agent, \
    plot_training_performance, evaluate_agent, plot_loss
import RL_experiments.parameters as params
from RL_experiments.standalone_simulatiion import Setup
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy




setup1 = Setup(**params.SETUP)


env           = RISEnv2(setup1, episode_length=10) #RIS_TFenv(config, 1, transmit_SNR=1)

train_env     = tf_py_environment.TFPyEnvironment(env)
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


num_eval_timesteps = 1000
random_policy_average_return = evaluate_agent(random_policy, train_env, num_eval_timesteps, name='Random')



agentParams = NeuralEpsilonGreedyParams(**params.NEURAL_EPSILON_GREEDY_PARAMS)
agent = initialize_NeuralEpsilonGreedyAgent(agentParams, train_env)


reward_values, loss_infos = train_bandit_agent(agent, train_env, agentParams)


plot_training_performance(reward_values, len(reward_values), 2*agentParams.log_interval, agent.name, random_policy_average_return)
plot_loss(reward_values, agent.name, scale='linear')


score = evaluate_agent(agent.policy, train_env, 20, agent.name)
score_as_percentage_of_random = (score / random_policy_average_return - 1) * 100
print(f'{agent.name} attained mean performance of {score} ( {score_as_percentage_of_random}% improvement of random policy).')