import numpy as np
import matplotlib.pyplot as plt

from utils.custom_configparser import CustomConfigParser
from RL_experiments.environments import RIS_TFenv, RISEnv2
from RL_experiments.training import initialize_DQN_agent, train_DQN_agent, DQNParams, plot_training_performance, \
    evaluate_agent, plot_loss

from RL_experiments.standalone_simulatiion import Setup

from tf_agents import utils
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy

import RL_experiments.parameters as params

# config = CustomConfigParser().load_from_file(open('setup_config.ini', 'r'))
# config.print()

# setup1        = Setup(K=1,
#                       S=10,
#                       M=1,
#                       N=4,
#                       group_size=1,
#                       RIS_positions=np.array([[5,25,2]]))

setup1 = Setup(**params.SETUP)

env           = RISEnv2(setup1, episode_length=10) #RIS_TFenv(config, 1, transmit_SNR=1)
train_env     = tf_py_environment.TFPyEnvironment(env)
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


num_eval_timesteps = 1000
random_policy_average_return = evaluate_agent(random_policy,
                                              train_env,
                                              num_eval_timesteps,
                                              name='Random')

num_actions = int(train_env.action_spec().maximum)-int(train_env.action_spec().minimum)+1


# dqn_params = DQNParams(
#     fc_layer_params              = (100, 100),
#     num_iterations               = 10*num_actions,
#     initial_collect_steps        = 1000,
#     collect_steps_per_iteration  = 1,
#     replay_buffer_max_length     = 10000,
#     batch_size                   = 32,
#     learning_rate                = 10e-3,
#     log_interval                 = 5,
#     eval_interval                = 10,
#     td_errors_loss_fn            = element_wise_squared_loss,
#     epsilon_greedy               = 0.2,
#     gradient_clipping            = 100,
#     n_step_update                = 1,
#     target_update_tau            = 1.0,
#     target_update_period         = 5,
#     gamma                        = 0.9999,
#     num_eval_episodes            = max(num_eval_timesteps // 10, 5)
# )


dqn_params = DQNParams(**params.DQN_PARAMS)

dqn_agent = initialize_DQN_agent(dqn_params, train_env)



dqn_rewards, dqn_losses = train_DQN_agent(dqn_agent, train_env, dqn_params, random_policy)

dqn_avg_score = evaluate_agent(dqn_agent.policy, train_env, num_eval_timesteps, 'DQN')

plot_training_performance(dqn_rewards, len(dqn_rewards), 10, 'DQN', random_policy_average_return)
plot_loss(dqn_losses, dqn_agent.name, scale='log')



score = evaluate_agent(dqn_agent.policy, train_env, 20, dqn_agent.name)
score_as_percentage_of_random = (1 - score / random_policy_average_return) * 100
print(f'{dqn_agent.name} attained mean performance of {score} ( {score_as_percentage_of_random}% improvement of random policy).')