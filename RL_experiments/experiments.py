import os
from collections import OrderedDict
from dataclasses import asdict

import matplotlib.pyplot as plt

from RL_experiments.train_OptimalAgent import OptimalAgent
from RL_experiments.train_RandomAgent import RandomAgent
from utils import notifyme
from utils.custom_types import VarsString
from utils.misc import split_filename_to_name_and_extension, generate_dirname

from utils.misc import lod_2_dol, smooth_curve

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd

import json
import os
from abc import ABC
from typing import Iterable, List, NewType, Dict

import numpy as np
from tqdm import tqdm
from termcolor import colored


from RL_experiments.environments import RISEnv2, find_best_action_exhaustively, evaluate_action, RISEnv3, RateRequestsEnv, RateQoSEnv
from RL_experiments.standalone_simulatiion import Setup
from RL_experiments.training_utils import Agent, compute_baseline_scores, display_and_save_results, plot_loss, \
    AgentParams

ALLOWED_OBSERVATION_TYPES = {'channels', 'angles'}



class Experiment:

    def __init__(self, stream_or_filename):

        if isinstance(stream_or_filename, str):
            f = open(stream_or_filename, 'r')
        else:
            f = stream_or_filename

        params = json.loads(f.read())
        f.close()

        self.setup_params                 = params['SETUP']
        self.exp_params                   = params['EXPERIMENT']
        self.setup                        = Setup(**self.setup_params)
        self.params                       = params

        self.num_iterations               = self.exp_params['n_iters']              # type: int
        self.num_eval_episodes            = self.exp_params['num_eval_episodes']    # type: int
        self.verbose                      = self.exp_params['verbose']              # type: bool
        self.num_evaluations              = self.exp_params['n_evals']              # type: int
        self.eval_interval                = self.exp_params['eval_interval']        # type: int

        self.plot_loss                    = self.exp_params['plot_loss']            # type: bool
        self.plot_performance             = self.exp_params['plot_performance']     # type: bool
        self.save_results                 = self.exp_params['save_results']         # type: bool
        self.plot_smooth_sigma            = self.exp_params['plot_smooth_sigma']    # type: float

        self.observation_type             = self.exp_params['observation_type']            # type: str
        self.num_cached_realizations      = self.exp_params['num_cached_realizations']     # type: int
        self.results_rootdir              = self.exp_params['results_rootdir']             # type: str
        self.baselines_performance_files  = self.exp_params['baselines_performance_files'] # type: str
        self.num_actions                  = None                                           # type: int

        self._generate_dirnames()

        self.observation_type = self.observation_type.lower()
        if self.observation_type not in ALLOWED_OBSERVATION_TYPES:
            raise ValueError(f"Unsupported observation type {self.observation_type}")



    def _generate_dirnames(self):
        all_params = {**self.params['SETUP'], **self.params['EXPERIMENT']}

        def get_create_dirname(path, varstring_name, prefix)->str:
            varstring = self.exp_params[varstring_name] if varstring_name else ""
            dirname   = generate_dirname(varstring, all_params, prefix=prefix)
            dirname   = os.path.join(path, dirname)
            os.makedirs(dirname, exist_ok=True)
            return dirname

        self.setup_dirname     = get_create_dirname(self.results_rootdir, "experiment_dir_vars", "Setup")
        self.data_dirname      = get_create_dirname(self.setup_dirname, "", self.exp_params['data_dirname'])
        baselines_rootdir      = get_create_dirname(self.setup_dirname, "", self.exp_params['baselines_rootdir'])
        agents_rootdir         = get_create_dirname(self.setup_dirname, "", self.exp_params['agents_rootdir'])
        self.baselines_dirname = get_create_dirname(baselines_rootdir, "baselines_dir_vars", "")
        self.agents_dirname    = get_create_dirname(agents_rootdir, "agent_dir_vars", "")

        self.exhaustive_baseline_dirname = os.path.join(self.baselines_dirname, 'Exhaustive')
        self.random_baseline_dirname     = os.path.join(self.baselines_dirname, 'Random')

    def _load_baseline_scores_from_saved_file(self):
        try:
            f = open(os.path.join(self.setup_dirname, "baselines.json"), "r")

            baseline_results = json.loads(f.read())

            self.optimal_score                = baseline_results['optimal_score']
            self.random_policy_average_return = baseline_results['random_policy_average_return']
            self.std_return                   = baseline_results['std_return']

    def _generate_agent_dirname(self, agent: Agent):
        agent_dirname = generate_dirname(agent.params.vals_in_dirname, asdict(agent.params), prefix=agent.name)
        agent_dirname = os.path.join(self.agents_dirname, agent_dirname)
        os.makedirs(agent_dirname, exist_ok=True)

        return agent_dirname



    def _check_baselines_exist(self):
        if not os.path.exists(self.exhaustive_baseline_dirname) or not os.path.exists(self.random_baseline_dirname):
            return False
        if len(os.listdir(self.exhaustive_baseline_dirname)) == 0 or len(os.listdir(self.random_baseline_dirname)) == 0:
            return False
        return True


    def _initialize_callbacks(self, env: RISEnv2,):
        comparisons = BaselinesComparison(env,
                                          os.path.join(self.exhaustive_baseline_dirname, self.baselines_performance_files),
                                          os.path.join(self.random_baseline_dirname, self.baselines_performance_files),
                                          verbose=1)
        history = HistoryCallback(env, save_observations=False)
        return comparisons, history


    def _update_agent_params(self, agentParams):
        agentParams.num_iterations    = self.num_iterations
        agentParams.num_eval_episodes = self.num_eval_episodes
        agentParams.num_evaluations   = self.num_evaluations
        agentParams.verbose           = self.verbose
        agentParams.eval_interval     = self.eval_interval


    def _initialize_agent(self, env, agent_or_agent_class, agent_params_class, agent_params_JSON_key, num_actions) -> Agent:
        if isinstance(agent_or_agent_class, Agent):
            agent = agent_or_agent_class
            self._update_agent_params(agent.params)
        else:
            agent_class = agent_or_agent_class
            agentParams = agent_params_class(**self.params[agent_params_JSON_key])
            self._update_agent_params(agentParams)
            agent = agent_class(agentParams, num_actions, env.observation_spec().shape[0], self.observation_type)  # type: Agent

        return agent


    def run(self, agent_or_agent_class, agent_params_class=None, agent_params_JSON_key=None):

        env                  = RISEnv3(self.setup, self.observation_type, self.num_cached_realizations, self.data_dirname)
        num_actions          = env.action_spec().maximum + 1
        self.num_iterations  = int(self.num_iterations * num_actions)

        if not self._check_baselines_exist():
            self.run_both_baselines(env)

        agent                = self._initialize_agent(env, agent_or_agent_class, agent_params_class, agent_params_JSON_key, num_actions)
        comparisons, history = self._initialize_callbacks(env)

        agent.train(env, [history], [comparisons])

        agent_results_dirname = self._generate_agent_dirname(agent)

        comparisons.save_scores_to_file(os.path.join(agent_results_dirname, "performance.csv"))
        history.save_scores_to_file(os.path.join(agent_results_dirname, "training.csv"))

        comparisons.plot(savefile=os.path.join(agent_results_dirname, "evaluation_performance.png"))
        history.plot(savefile=os.path.join(agent_results_dirname,    "training_loss_and_reward.png"))

        notifyme.send_notification("Agent finished")

        return agent, None



    def run_baseline(self, env, baseline_agent_class):
        if env is None:
            env = RISEnv3(self.setup, self.observation_type, self.num_cached_realizations, self.data_dirname)

        num_actions    = env.action_spec().maximum + 1
        baseline_agent = self._initialize_agent(env, baseline_agent_class, AgentParams, "DUMMY_PARAMS", num_actions)

        if baseline_agent.name == 'OptimalAgent': dirname = 'Exhaustive'
        elif baseline_agent.name == 'RandomAgent': dirname = 'Random'
        else: raise ValueError


        cb1 = EvaluationSaverCallback(env, verbose=1)
        baseline_agent.train(env, [], [cb1])

        baseline_scores_file = os.path.join(self.baselines_dirname, dirname, self.baselines_performance_files)
        os.makedirs(os.path.join(self.baselines_dirname, dirname))
        cb1.save_scores_to_file(baseline_scores_file)

        return baseline_agent, None



    def run_both_baselines(self, env):
        if self.verbose:
            print("\n\n=== Running Baselines ===\n")

        self.run_baseline(env, RandomAgent)
        self.run_baseline(env, OptimalAgent)








class TrainingCallback:
    def __init__(self, name, env: RISEnv2, verbose: int):
        self.name = name
        self.env  = env
        self.verbose = verbose

    def __call__(self, step, obs=None, action=None, reward=None):
        raise NotImplementedError

    def save_scores_to_file(self, filename, **kwargs)->None:
        raise NotImplementedError

    def load_scores_from_file(self, filename, **kwargs)->pd.DataFrame:
        raise NotImplementedError


class EvaluationSaverCallback(TrainingCallback):
    def __init__(self, env, verbose):
        super(EvaluationSaverCallback, self).__init__("Evaluation", env, verbose)
        self.eval_time_steps = OrderedDict()


    def __call__(self, step=None, obs=None, action=None, reward=None, **kwargs):
        if reward is None or step is None: raise ValueError
        step, reward = int(step) , float(reward)

        if step in self.eval_time_steps.keys(): raise ValueError(f"Time step '{step}' already has an evaluation reward saved.")

        self.eval_time_steps[step] = reward

        if self.verbose > 0:
            tqdm.write(f"[{self.name}] {step:04d} : Reward: {reward}")


    def save_scores_to_file(self, filename, mode=None, **kwargs) ->None:
        data_dict = {
            'Time Step': self.eval_time_steps.keys(),
            'Reward'   : self.eval_time_steps.values(),
        }

        df = pd.DataFrame(data=data_dict)
        df = df.set_index('Time Step', drop=True)

        # if mode == 'append':
        #     filename = self._increment_filename_counter(filename)

        df.to_csv(filename)


    def load_scores_from_file(self, filename, **kwargs) ->pd.DataFrame:
        df = pd.read_csv(filename)
        if 'Time Step' not in df.columns or 'Reward' not in df.columns: raise ValueError

        self.eval_time_steps = OrderedDict(zip(df['Time Step'], df['Reward']))

        return df.set_index('Time Step', drop=True)




class BaselinesComparison(TrainingCallback):
    def __init__(self, env, optimal_base_filename: str, random_base_filename: str, verbose: int):
        super(BaselinesComparison, self).__init__("Baselines Comparison", env, verbose)
        self.time_steps = []
        self.rewards    = []

        self.average_optimal_evaluations = self._load_baseline_scores_from_file(optimal_base_filename)
        self.average_random_evaluations  = self._load_baseline_scores_from_file(random_base_filename)

        self.fraction_upper = None
        self.fraction_lower = None

        self.df = None


    def _load_baseline_scores_from_file(self, filename):
        try:
            df = pd.read_csv(filename)
            return OrderedDict(zip(df['Time Step'], df['Reward']))
        except (KeyError, FileNotFoundError) as e:
            raise EnvironmentError(f"Baseline file '{filename}' does not exist or corrupted.")



    def __call__(self, step=None, obs=None, action=None, reward=None, **kwargs):
        if reward is None: raise ValueError

        step, reward = int(step), float(reward)

        self.time_steps.append(step)
        self.rewards.append(reward)


        try:
            r_max_mean = self.average_optimal_evaluations[step]
            r_rand_mean = self.average_random_evaluations[step]
        except KeyError:
            raise ValueError(f"Step {step} does not exist in stored baseline dict(s)!")

        reward_frac_optimal = reward / r_max_mean
        reward_frac_random = reward / r_rand_mean



        if self.verbose > 0:

            if self.fraction_upper:
                color = "green" if reward >= self.fraction_upper * r_max_mean else "red"
                reward_frac_optimal = colored(f"{reward_frac_optimal:.4f}", color)
            else:
                reward_frac_optimal = f"{reward_frac_optimal:.4f}"

            if self.fraction_lower:
                color = "green" if reward >= self.fraction_lower * r_rand_mean else "red"
                reward_frac_random = colored(f"{reward_frac_random:.4f}", color)
            else:
                reward_frac_random = f"{reward_frac_random:.4f}"

            tqdm.write(
                f"[{self.name}]: {step:04d} | Achieved reward: {reward:.4f} ( {reward_frac_random} of random, {reward_frac_optimal} of optimal ).")




    def save_scores_to_file(self, filename, **kwargs):


        if len(self.average_optimal_evaluations.keys()) != len(self.time_steps) or len(self.average_random_evaluations.keys()) != len(self.time_steps):
            raise ValueError

        rewards = np.array(self.rewards)



        data_dict = {
            'Time Step'           : self.time_steps,
            'Reward'              : rewards,
            'Optimal Score'       : self.average_optimal_evaluations.values(),
            'Random Score'        : self.average_random_evaluations.values(),
            'rewards_wrt_optimal' : rewards / np.array(list(self.average_optimal_evaluations.values())),
            'rewards_wrt_random'  : rewards / np.array(list(self.average_random_evaluations.values())),
        }

        df = pd.DataFrame(data=data_dict)
        #df = df.set_index('Time Step', drop=True)

        df.to_csv(filename, index=False)
        self.df = df


    def load_scores_from_file(self, filename, **kwargs):

        df = pd.read_csv(filename)
        self.df = df

        return df

    def plot(self, agent_name=None, savefile=None):


        if agent_name is None: agent_name = "method"

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        ax.plot(self.df['Time Step'], self.df['Reward'], '-r', label=agent_name)
        ax.plot(self.df['Time Step'], self.df['Optimal Score'], '--k', label="optimal")
        ax.plot(self.df['Time Step'], self.df['Random Score'], '-.b', label="random")
        ax.set_ylabel('Reward')
        ax.set_xlabel('time steps')
        plt.legend()
        plt.title(f"Evaluation of performance of {agent_name}")

        if savefile is not None:
            plt.savefig(savefile)

        try:
            plt.show(block=False)
        except Exception as e:
            print(f"Warning: Exception during showing figure! \n\n{e}\n")

        random_as_fraction_of_optimal = self.df['Random Score'] / self.df['Optimal Score']

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        ax.plot(self.df['Time Step'], self.df['rewards_wrt_optimal'], '-r', label=agent_name)
        ax.plot(self.df['Time Step'], random_as_fraction_of_optimal, '--k', label="random")
        ax.set_ylabel('Normalized reward')
        ax.set_xlabel('time steps')
        plt.legend()
        plt.title("Evaluation of performance of {agent_name} normalized by optimal")

        if savefile is not None:
            fname_parts = savefile.split('.')
            fname_no_ext = ".".join(fname_parts[:-1])
            fname_ext = fname_parts[-1]

            savefile = fname_no_ext + "_normalized" + "." + fname_ext
            plt.savefig(savefile)

        try:
            plt.show(block=False)
        except Exception as e:
            print(f"Warning: Exception during showing figure! \n\n{e}\n")




class HistoryCallback(TrainingCallback):

    def __init__(self, env, save_observations):
        super(HistoryCallback, self).__init__("History", env, verbose=0)
        self.save_observations = save_observations
        self.steps        = []
        self.observations = []
        self.actions      = []
        self.rewards      = []
        self.losses       = []

        ################# K ###################
        self.infos = []

    def __call__(self, step, obs=None, action=None, reward=None, loss=None, **kwargs):
        self.steps.append(int(step))
        self.observations.append(obs if self.save_observations else None)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        ################# K ###################
        self.infos.append(info)
        if loss is not None:
            #print('[debug] saving loss')
            try:
                self.losses.append(float(loss))
            except TypeError:
                if not loss:
                    self.losses.append(None)
                else:
                    try:
                        self.losses += loss
                    except TypeError:
                        raise ValueError("Unexpected value for loss. Expected float-like or list")
        else:
            #print('[debug] NOT saving loss. '+f"kwargs: {kwargs.keys()}")
            pass



        return False # never converges due to this function

    def save_scores_to_file(self, filename, **kwargs) -> None:
        data_dict = {
            'Time Step'      : self.steps,
            'Observation'    : self.observations,
            'Action'         : self.actions,
            'Reward'         : self.rewards,
        }

        if len(self.losses) == len(self.steps):
            data_dict['Loss'] = self.losses

        df = pd.DataFrame(data=data_dict)
        df = df.set_index('Time Step', drop=True)


        df.to_csv(filename)


    def load_scores_from_file(self, filename, **kwargs) -> pd.DataFrame:

        df = pd.read_csv(filename, index_col='Time Step')

        self.steps        = list(df.index)
        self.observations = list(df['Observation'])
        self.actions      = list(df['Action'])
        self.rewards      = list(df['Reward'])

        if 'Loss' in df.columns:
            self.losses = list(df['Loss'])

        return df



    def plot(self, agent_name=None, savefile=None):

        fig, ax = plt.subplots(figsize=(18,12))
        ax.plot(self.steps, self.rewards, 'r')
        ax.set_ylabel('Reward', color='r')
        ax.set_xlabel('time steps')


        losses = np.array(self.losses, dtype=float)
        if len(losses) > 0 and len(losses) <= len(self.steps):
            steps = np.array(self.steps, dtype=int)

            idx_not_nan_losses = np.argwhere(1 - np.isnan(losses))

            losses = losses[idx_not_nan_losses]
            steps  = steps[idx_not_nan_losses]

            ax2 = ax.twinx()
            ax2.plot(steps, losses, 'b')
            ax2.set_ylabel('Loss', color='b')



        if agent_name is None: agent_name = "the method"
        plt.title(f"Training performance of {agent_name}")

        if savefile is not None:
            plt.savefig(savefile)

        try:
            plt.show(block=False)
        except Exception as e:
            print(f"Warning: Exception during showing figure! \n\n{e}\n")




class RewardMonitoringCallback(ConvergenceCallback):
    def __init__(self, frequency=50, filename=None, verbose=True):
        super(RewardMonitoringCallback, self).__init__("Exploration Evaluation")
        self.frequency = frequency
        self.ts = 0
        self.verbose = verbose
        self.rewards = np.zeros(self.frequency)
        self.filename = filename



    def __call__(self,  step, obs=None, action=None, reward=None, **kwargs):
        if reward is None: raise ValueError

        self.rewards[self.ts] = reward
        self.ts = (self.ts + 1 ) % self.frequency

        if self.ts == 0:

            if self.verbose:
                tqdm.write(f"> Average reward: {self.rewards.mean():.3f}")

            # with open(self.filename, "a") as fout:
            #     fout.write("\n".join(map(str, self.rewards)))
            #     fout.write("\n")


        return False




################# K ###################
def plot_rate_requests_penalties(training_info: List[Dict], env: RateRequestsEnv):
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap


    info = lod_2_dol(training_info[-300:-10], numpy=True)

    colors = get_cmap('Pastel1').colors

    fig = plt.figure()
    ys  = np.empty_like(info['requests_penalties'])

    for i in range(env.setup.K):
        y = info['requests_penalties'][:,i]
        y = env.setup.rate_requests[i] + y
        y = smooth_curve(y, 50, 5)
        plt.plot(info['t'], y, label=f'UE {i}', c=colors[i])
        plt.hlines(y=env.setup.rate_requests[i], xmin=info['t'][0], xmax=info['t'][-1], linestyles='--', color=colors[i])
        ys[:,i] = y


    ys_avg = ys.mean(axis=1)
    plt.plot(info['t'], ys_avg, c='k')

    plt.xlabel('Time step')
    plt.ylabel('SINR per user')
    plt.legend()
    plt.show(block=False)
