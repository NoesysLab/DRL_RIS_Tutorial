import json
import os
from copy import deepcopy
from typing import Tuple, Callable, Union
import numpy as np
import matplotlib.pyplot as plt
import builtins
from dataclasses import dataclass, field, asdict
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
from tqdm import tqdm
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from RL_experiments.environments import RISEnv2
from RL_experiments.standalone_simulatiion import Setup

# ------------------------------- EVALUATION FUNCTIONS ----------------------------- #
from RL_experiments.environments import RISEnv2, compute_average_optimal_policy_return
from RL_experiments.standalone_simulatiion import Setup
from utils.notifyme import send_notification


@dataclass
class AgentParams:
    num_iterations    : int
    num_evaluations   : int
    num_eval_episodes : int

    def __post_init__(self):
        pass


class Agent:

    def __init__(self, name: str, params: AgentParams, num_actions, observation_dim):
        self.name            = name
        self.params          = params
        self.num_actions     = num_actions
        self.observation_dim = observation_dim

    @property
    def policy(self):
        return None

    @property
    def collect_policy(self):
        return None

    def train(self, env):
        raise ValueError


def run_experiment(params_filename, agent_class, agent_params_class, agent_params_JSON_key, agent_params_in_dirname):


    params                          = json.loads(open(params_filename).read())
    setup1                          = Setup(**params['SETUP'])
    agentParams                     = agent_params_class(**params[agent_params_JSON_key])
    env                             = RISEnv2(setup1, episode_length=np.inf)
    agent                           = agent_class(agentParams,
                                                  env.action_spec().maximum + 1,
                                                  env.observation_spec().shape[0])
    optimal_score,\
    random_policy_average_return,\
    std_return                      = compute_baseline_scores(env, setup1, agentParams, print_=True)

    reward_values,\
    losses,\
    eval_steps,\
    best_policy                     = agent.train(env)

    avg_score,\
    std_return                      = evaluate_agent(agent, env)

    display_and_save_results(agent, params, agentParams, random_policy_average_return, optimal_score, avg_score,
                             std_return, reward_values, eval_steps, smooth_sigma=5, agent_params_in_dirname=agent_params_in_dirname)















def compute_avg_return(environment, policy, num_timesteps=100):
    """
    :param environment: A  TfPyEnvironment instance
    :param policy: A TFPolicy in
    :param num_timesteps:
    :return: float
    """
    returns = []
    time_step = environment.reset()

    for ts_counter in range(num_timesteps):
        if  time_step.is_last():
            time_step = environment.reset()

        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        returns.append( float(time_step.reward.numpy()) )


    returns = np.array(returns)
    avg_return = float(np.mean(returns))
    std_return = float(np.std(returns))

    return avg_return, std_return




def evaluate_agent(agent, env: RISEnv2):

    rewards = np.empty(agent.params.num_eval_episodes)
    time_step = env._reset()

    for i in range(agent.params.num_eval_episodes):
        if time_step.is_last(): time_step = env._reset()

        obs        = time_step.observation
        action     = agent.policy(obs)
        time_step  = env._step(action)
        reward     = time_step.reward
        rewards[i] = reward

    return rewards.mean(), rewards.std()



# -------------- PLOTTING FUNCTIONS ---------------------



def plot_loss(loss_values, agent_name, scale='linear', figsize=(16,9), smooth_sigma=None):
    plt.figure(figsize=figsize)
    x = np.arange(1, len(loss_values)+1)
    plt.plot(x, loss_values, label='original values')

    if smooth_sigma is not None:
        ysmoothed = gaussian_filter1d(loss_values, sigma=smooth_sigma)
        plt.plot(x, ysmoothed, label='smoothed')
        plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel('Train loss')
    plt.yscale(scale)
    plt.title(f'{agent_name} training loss')
    plt.show()


def plot_training_performance(reward_values, iteration_timesteps, name=None, random_avg_reward=None, optimal_avg_reward=None, smooth_sigma=None):
    sns.set_theme()

    name = name if name is not None else 'Trained agent'


    plt.plot(iteration_timesteps, reward_values, alpha=.7, label=name)

    if random_avg_reward is not None:
        plt.hlines([random_avg_reward], 0, iteration_timesteps[-1], color='grey', ls=':', label='random policy')

    if optimal_avg_reward is not None:
        plt.hlines([optimal_avg_reward], 0, iteration_timesteps[-1], color='k', ls='--', label='optimal policy')


    if smooth_sigma is not None:
        ysmoothed = gaussian_filter1d(reward_values, sigma=smooth_sigma)
        plt.plot(iteration_timesteps, ysmoothed, label=f'{name} (smoothed)')

    plt.legend()

    plt.ylabel('Reward')
    plt.xlabel('Number of Iterations')
    plt.show(block=False)





def save_results(agent_name            : str,
                 setupParams           : dict,
                 agentParams           : dict,
                 reward_list           : list,
                 eval_steps            : list,
                 results_dict          : dict,
                 setup_dirname_params  : str,
                 agent_dirname_params  : str,
                 results_rootdir        = './results/',
                 ):

    def to_format_string(s):
        out = ''
        for variable in s.split(','):
            out += "_" + variable + "_{" + variable +"}"
        return out

    def generate_dirname(dirname_params, values_dict, prefix=''):
        fstring = to_format_string(dirname_params)
        dirname = fstring.format(**values_dict)
        if prefix:
            dirname = prefix + "_" + dirname
        return dirname + "/"

    setup_dirname = generate_dirname(setup_dirname_params, setupParams, prefix='setup')
    agent_dirname = generate_dirname(agent_dirname_params, agentParams, prefix=agent_name)

    all_dirs = os.path.join(results_rootdir, setup_dirname, agent_dirname)
    os.makedirs(all_dirs, exist_ok=True)

    with open(os.path.join(results_rootdir, setup_dirname, 'setup.json'), 'w') as fout:
        fout.write(json.dumps(setupParams, indent=4), )

    with open(os.path.join(results_rootdir, setup_dirname, agent_dirname, 'agent_params.json'), 'w') as fout:
        fout.write(json.dumps(agentParams, indent=4))

    with open(os.path.join(results_rootdir, setup_dirname, agent_dirname, 'agent_performance.json'), 'w') as fout:
        fout.write(json.dumps(results_dict, indent=4))

    with open(os.path.join(results_rootdir, setup_dirname, agent_dirname, 'agent_training.csv'), 'w') as fout:
        pd.DataFrame({
            'iteration' : eval_steps,
            'reward'    : reward_list,
        }).to_csv(fout, index=False)






def cond_print(print_, *args, **kwargs):
    if print_:
        print(*args, **kwargs)




def compute_baseline_scores(env: RISEnv2, setup: Setup, agentParams: dataclass, print_=True):
    cond_print(print_, f"\nRunning with {env.action_spec().maximum + 1} actions ({setup.N_controllable} bits for RIS configurations, {env.codebook_size_bits_required} bits for codebook).")
    cond_print(print_, f"Observation space is of dimension: {env.observation_spec().shape}.")

    train_env = tf_py_environment.TFPyEnvironment(env)
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    optimal_score = compute_average_optimal_policy_return(env, timesteps=agentParams.num_eval_episodes)
    cond_print(print_, f"Score of optimal policy: {optimal_score}\n")

    random_policy_average_return, std_return = compute_avg_return(train_env, random_policy,
                                                                  agentParams.num_eval_episodes)
    cond_print(print_, f"\nRandom policy average return: {random_policy_average_return} +/- {std_return:3f}\n")

    return optimal_score, random_policy_average_return, std_return



def display_and_save_results(agent,
                             params,
                             agent_params,
                             random_policy_average_return,
                             optimal_score,
                             avg_score,
                             std_return,
                             reward_values,
                             eval_steps,
                             smooth_sigma=5,
                             agent_params_in_dirname="num_iterations,learning_rate"):

    plot_training_performance(reward_values, eval_steps, agent.name, random_policy_average_return, optimal_score,
                              smooth_sigma=smooth_sigma)

    score_as_percentage_of_random = (avg_score / random_policy_average_return - 1) * 100
    print(
        f'{agent.name} attained mean performance of {avg_score} +/- {std_return:.3f} ( {score_as_percentage_of_random}% improvement of random policy).')

    score_as_percentage_of_optimal = (avg_score / optimal_score) * 100
    print(f'Achieved performance is {score_as_percentage_of_optimal} of average optimal policy.')

    print("Saving results...")

    save_results(agent.name,
                 params['SETUP'],
                 asdict(agent_params),
                 reward_values,
                 eval_steps,
                 {
                     "avg_score": avg_score,
                     "std_return": std_return,
                     "random_policy_average_return": random_policy_average_return,
                     "score_as_percentage_of_random": score_as_percentage_of_random,
                     "score_as_percentage_of_optimal": score_as_percentage_of_optimal
                 },
                 "N_controllable,K,M,codebook_rays_per_RX,kappa_H,observation_noise_variance",
                 agent_params_in_dirname
                 )

    send_notification(
        f"{agent.name} finished. \n {score_as_percentage_of_random}% above random\n {score_as_percentage_of_optimal}% of optimal")