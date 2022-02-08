import os
from abc import ABC

from utils.custom_types import VarsString
from utils.misc import cond_print

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


import json
import os
from copy import deepcopy
from typing import Tuple, Callable, Union, Iterable, List
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



def tqdm_(iterator, *args, verbose_=True, **kwargs,):
    if verbose_:
        return tqdm(iterator, *args, **kwargs)
    else:
        return iterator

@dataclass
class AgentParams:
    num_iterations    : int         = None
    num_evaluations   : int         = None
    eval_interval     : int         = None
    num_eval_episodes : int         = None
    verbose           : bool        = None
    vals_in_dirname   : VarsString  = None

    def __post_init__(self):
        if self.eval_interval is not None and self.num_evaluations is not None:
            raise ValueError("Conflicting arguments: Expected one out of `num_evaluations`, `eval_interval` not to be null, found values in both.")


class Agent:

    def __init__(self, name: str, params: AgentParams, num_actions, observation_dim, observation_type):
        self.name             = name
        self.params           = params
        self.num_actions      = num_actions
        self.observation_dim  = observation_dim
        self.observation_type = observation_type

        if self.params.eval_interval is None:
            self.eval_interval = self.params.num_iterations // self.params.num_evaluations
        else:
            self.eval_interval = self.params.eval_interval

    @property
    def policy(self):
        raise NotImplemented

    @property
    def collect_policy(self):
        raise NotImplemented

    def _initialize_training_vars(self, env: RISEnv2):
        raise NotImplementedError

    def _apply_collect_step(self, step, obs, action, reward)->None:
        raise NotImplementedError

    def _perform_update_step(self)->List:
        raise NotImplementedError



    def train(self, env: RISEnv2, training_callbacks=None, eval_callbacks=None):
        if training_callbacks is None: training_callbacks = []
        if eval_callbacks is None: eval_callbacks = []



        self._initialize_training_vars(env)

        rewards      = []
        reward_steps = []
        losses       = []

        #initial_reward, _ = self.evaluate(env)
        #rewards.append(initial_reward)
        #reward_steps.append(0)

        if self.params.verbose: print('Starting training')

        time_step = env._reset()

        try:
            for step in tqdm(range(self.params.num_iterations)):

                if time_step.is_last():
                    time_step = env._reset()



                obs       = time_step.observation
                action    = self.collect_policy(obs)
                time_step = env._step(action)
                reward    = time_step.reward

                self._apply_collect_step(step, obs, action, reward)

                this_step_losses = self._perform_update_step()
                losses += this_step_losses

                #time_step = next_time_step


                if step % self.eval_interval == 0:
                    avg_score, std_score = self.evaluate(env)
                    #tqdm.write(f"step={step} | Avg reward = {avg_score} +/- {std_score}.")
                    rewards.append(avg_score)
                    reward_steps.append(step)

                    converged_flag, converged_callback_names = apply_callbacks(eval_callbacks, step, reward=avg_score)
                    if converged_flag:
                        tqdm.write(f"Step={step} | Algorithm converged due to criteria: {converged_callback_names}")
                        break


                converged_flag, converged_callback_names = apply_callbacks(training_callbacks, step, obs, action, reward, info=env.get_info(), loss=this_step_losses)
                if converged_flag:
                    tqdm.write(f"Step={step} | Algorithm converged due to criteria: {converged_callback_names}")
                    break

        except KeyboardInterrupt:
            print("Training stopped by user...")

        return rewards, losses, reward_steps, self.policy





    def evaluate(self, env: RISEnv2, return_info=False, n_iters=None, verbose=False):

        n_iters = self.params.num_eval_episodes if n_iters is None else n_iters

        rewards = np.empty((n_iters,))
        #time_step = env._reset()
        time_step = env.current_time_step()

        info = {'observation': [], 'action': [], 'reward': []}


        iter = range(n_iters) if not verbose else tqdm(range(n_iters))
        for i in iter:
            if time_step.is_last():
                time_step = env._reset()

            obs = time_step.observation
            action = self.policy(obs)
            time_step = env._step(action)
            reward = time_step.reward
            rewards[i] = reward

            if return_info:
                info['observation'].append(obs)
                info['action'].append(action)
                info['reward'].append(reward)

        if n_iters == 1:
            if return_info: return rewards[0], info
            else: return rewards[0]

        else:
            if return_info:
                return rewards.mean(), rewards.std(), info
            else:
                return rewards.mean(), rewards.std()











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







# -------------- PLOTTING FUNCTIONS ---------------------



def plot_loss(loss_values, agent_name, scale='linear', figsize=(16,9), smooth_sigma=None, savefile=None):
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

    if savefile is not None:
        plt.savefig(savefile)

    try:
        plt.show(block=False)
    except Exception as e:
        print(f"Warning: Exception during showing figure! \n\n{e}\n")


def plot_training_performance(reward_values, iteration_timesteps, name=None, random_avg_reward=None, optimal_avg_reward=None, smooth_sigma=None, savefile=None):
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

    if savefile is not None:
        plt.savefig(savefile)

    try:
        plt.show(block=False)
    except Exception as e:
        print(f"Warning: Exception during showing figure! \n\n{e}\n")





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











def compute_baseline_scores(env: RISEnv2, setup: Setup, num_eval_episodes: int, print_=True):
    cond_print(print_, f"\nRunning with {env.action_spec().maximum + 1} actions ({setup.N_controllable} bits for RIS configurations, {env.codebook_size_bits_required} bits for codebook).")
    cond_print(print_, f"Observation space is of dimension: {env.observation_spec().shape}.")

    train_env = tf_py_environment.TFPyEnvironment(env)
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    optimal_score = compute_average_optimal_policy_return(env, timesteps=num_eval_episodes)
    cond_print(print_, f"Score of optimal policy: {optimal_score}\n")

    random_policy_average_return, std_return = compute_avg_return(train_env, random_policy, num_eval_episodes)
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
                             losses,
                             smooth_sigma=10,
                             agent_params_in_dirname="num_iterations,learning_rate",
                             results_rootdir='./results/'):


    plot_loss(losses, agent.name)

    plot_training_performance(reward_values, eval_steps, agent.name, random_policy_average_return, optimal_score,
                              smooth_sigma=smooth_sigma)

    score_as_percentage_of_random = (avg_score / random_policy_average_return - 1) * 100
    cond_print(agent_params.verbose, f'{agent.name} attained mean performance of {avg_score} +/- {std_return:.3f} ( {score_as_percentage_of_random}% improvement of random policy).')

    score_as_percentage_of_optimal = (avg_score / optimal_score) * 100
    cond_print(agent_params.verbose, f'Achieved performance is {score_as_percentage_of_optimal} of average optimal policy.')

    cond_print(agent_params.verbose, "Saving results...")

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
                 agent_params_in_dirname,
                 results_rootdir
                 )

    send_notification(
        f"{agent.name} finished. \n {score_as_percentage_of_random}% above random\n {score_as_percentage_of_optimal}% of optimal")




def apply_callbacks(callbacks: Iterable, step, obs=None, action=None, reward=None, **kwargs):
    converged          = False
    converged_cb_names = []
    for cb in callbacks:
        convergence_flag = cb(step, obs, action, reward, **kwargs)
        converged = converged or convergence_flag
        if convergence_flag:
            converged_cb_names.append(cb.name)

    converged_cb_names_str = ", ".join(converged_cb_names)
    return converged, converged_cb_names_str


