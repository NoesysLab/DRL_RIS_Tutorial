import sys

import numpy as np
import matplotlib.pyplot as plt
import json

import os

from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from RL_experiments.test_environment import TestEnv
from utils.custom_configparser import CustomConfigParser
from RL_experiments.environments import RISEnv2, compute_average_optimal_policy_return
from RL_experiments.training import NeuralEpsilonGreedyAgent, NeuralEpsilonGreedyParams, \
    initialize_NeuralEpsilonGreedyAgent, train_bandit_agent, \
    plot_training_performance, evaluate_agent, plot_loss, compute_avg_return, save_results
from RL_experiments.standalone_simulatiion import Setup
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy

from utils.notifyme import send_notification


params = json.loads(open(sys.argv[1]).read())

setup1 = Setup(**params['SETUP'])
agentParams = NeuralEpsilonGreedyParams(**params['NEURAL_EPSILON_GREEDY_PARAMS'])


env = RISEnv2(setup1, episode_length=np.inf) #RIS_TFenv(config, 1, transmit_SNR=1)




print(f"\nRunning with {env.action_spec().maximum +1} actions ({setup1.N_controllable} bits for RIS configurations, {env.codebook_size_bits_required} bits for codebook).")
print(f"Observation space is of dimension: {env.observation_spec().shape}." )


train_env     = tf_py_environment.TFPyEnvironment(env)
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


optimal_score = compute_average_optimal_policy_return(env, timesteps=500)
print(f"Score of optimal policy: {optimal_score}\n")


agent = initialize_NeuralEpsilonGreedyAgent(agentParams, train_env)


#random_policy_average_return = evaluate_agent(random_policy, train_env, num_eval_timesteps, name='Random')
random_policy_average_return, std_return = compute_avg_return(train_env, random_policy, agentParams.num_eval_episodes)
print(f"\nRandom policy average return: {random_policy_average_return} +/- {std_return:3f}\n")


reward_values, _, eval_steps, best_policy = train_bandit_agent(agent, train_env, agentParams)


plot_training_performance(reward_values, len(reward_values), 2*agentParams.log_interval, agent.name, random_policy_average_return, smooth_sigma=5)
#plot_loss(reward_values, agent.name, scale='linear', smooth_sigma=1.5)


avg_score, std_return = compute_avg_return(train_env, best_policy, agentParams.num_eval_episodes*10)
score_as_percentage_of_random = (avg_score / random_policy_average_return - 1) * 100
print(f'{agent.name} attained mean performance of {avg_score} +/- {std_return:.3f} ( {score_as_percentage_of_random}% improvement of random policy).')


avg_training_performance = np.array(reward_values).mean()
training_score_as_percentage_of_random = (avg_training_performance/random_policy_average_return-1)*100
print(f'{agent.name} attained mean performance of {avg_training_performance} ( {training_score_as_percentage_of_random}% improvement of random policy).')


send_notification(f"{agent.name} finished.")


print("Saving results...")

save_results(agent.name,
             params['SETUP'],
             params['NEURAL_EPSILON_GREEDY_PARAMS'],
             reward_values,
             eval_steps,
             avg_score,
             std_return,
             random_policy_average_return,
             score_as_percentage_of_random,
             training_score_as_percentage_of_random,
             "N_controllable,K,M,codebook_rays_per_RX",
             "num_iterations,learning_rate,learning_rate,gradient_clipping"
             )