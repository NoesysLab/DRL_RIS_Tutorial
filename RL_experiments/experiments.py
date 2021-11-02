import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


import json
import os
from abc import ABC
from typing import Iterable, List

import numpy as np
from tqdm import tqdm
from termcolor import colored


from RL_experiments.environments import RISEnv2
from RL_experiments.standalone_simulatiion import Setup
from RL_experiments.training_utils import Agent, compute_baseline_scores, display_and_save_results


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

        self.num_iterations               = self.exp_params['num_iterations']       # type: int
        self.num_eval_episodes            = self.exp_params['num_eval_episodes']    # type: int
        self.verbose                      = self.exp_params['verbose']              # type: bool
        self.num_evaluations              = self.exp_params['num_evaluations']      # type: int
        self.plot_loss                    = self.exp_params['plot_loss']            # type: bool
        self.plot_performance             = self.exp_params['plot_performance']     # type: bool
        self.save_results                 = self.exp_params['save_results']         # type: bool
        self.plot_smooth_sigma            = self.exp_params['plot_smooth_sigma']    # type: float
        self.results_rootdir              = self.exp_params['results_rootdir']      # type: str
        self.vars_in_dirname              = self.exp_params['vars_in_dirname']      # type: str
        self.use_cached_baselines         = self.exp_params['use_cached_baselines'] # type: bool
        self.eval_interval                = self.exp_params['eval_interval']        # type: int

        self.num_actions                  = None                                    # type: int
        self.optimal_score                = None                                    # type: float
        self.random_policy_average_return = None                                    # type: float
        self.std_return                   = None                                    # type: float

        self.setup_dirname                = self._generate_setup_dirname()
        self.training_callbacks           = []                                      # type: List[TrainingCallback]
        self.eval_callbacks               = []                                      # type: List[TrainingCallback]

    def _generate_setup_dirname(self) -> str:
        def to_format_string(s):
            out = ''
            for variable in s.split(','):
                out += "_" + variable + "_{" + variable + "}"
            return out

        def generate_dirname(dirname_params, values_dict, prefix=''):
            fstring = to_format_string(dirname_params)
            dirname = fstring.format(**values_dict)
            if prefix:
                dirname = prefix + "_" + dirname
            return dirname + "/"

        setup_dirname = generate_dirname(self.exp_params['vars_in_dirname'], self.setup_params, prefix='setup')
        setup_dirname = os.path.join(self.results_rootdir, setup_dirname)
        os.makedirs(setup_dirname, exist_ok=True)

        return setup_dirname

    def _load_baseline_scores_from_saved_file(self):
        try:
            f = open(os.path.join(self.setup_dirname, "baselines.json"), "r")

            baseline_results = json.loads(f.read())

            self.optimal_score                = baseline_results['optimal_score']
            self.random_policy_average_return = baseline_results['random_policy_average_return']
            self.std_return                   = baseline_results['std_return']

            f.close()
            return True
        except FileNotFoundError:
            return False

        except KeyError:
            self.optimal_score, self.random_policy_average_return, self.std_return = None, None, None
            return False

    def _store_baseline_scores_to_file(self):

        filename = os.path.join(self.setup_dirname, "baselines.json")

        with open(filename, "w") as f:
            f.write(json.dumps({
                'optimal_score' : self.optimal_score,
                'random_policy_average_return': self.random_policy_average_return,
                'std_return' : self.std_return
            }, indent=4))


    def _initialize_callbacks(self):

        self.training_callbacks = []
        self.eval_callbacks     = []

        for callback_dict in self.exp_params["convergence"]:
            callback_name   = next(iter(callback_dict))
            callback_params = callback_dict[callback_name]
            cb_name         = callback_name.lower().replace(' ', '_')

            if cb_name == 'upper_bound_percentage':
                self.eval_callbacks.append(UpperBoundPercentageConvergence(self.optimal_score, **callback_params))

        self.training_callbacks.append(HistoryCallback())



    def _update_agent_params(self, agentParams):
        agentParams.num_iterations    = self.num_iterations
        agentParams.num_eval_episodes = self.num_eval_episodes
        agentParams.num_evaluations   = self.num_evaluations
        agentParams.verbose           = self.verbose
        agentParams.eval_interval     = self.eval_interval





    def run(self, agent_or_agent_class, agent_params_class=None, agent_params_JSON_key=None):


        env                 = RISEnv2(self.setup, episode_length=np.inf)
        num_actions         = env.action_spec().maximum + 1
        self.num_iterations = int(self.num_iterations * num_actions)

        if isinstance(agent_or_agent_class, Agent):
            agent = agent_or_agent_class
            self._update_agent_params(agent.params)
        else:
            agent_class = agent_or_agent_class
            agentParams = agent_params_class(**self.params[agent_params_JSON_key])
            self._update_agent_params(agentParams)
            agent = agent_class(agentParams, num_actions, env.observation_spec().shape[0])  # type: Agent




        if not self.use_cached_baselines or not self._load_baseline_scores_from_saved_file():
            self.optimal_score, self.random_policy_average_return, self.std_return = compute_baseline_scores(env, self.setup, self.num_eval_episodes, print_=self.verbose)
            #self._store_baseline_scores_to_file()


        self._initialize_callbacks()


        reward_values, losses, eval_steps, best_policy = agent.train(env, self.training_callbacks, self.eval_callbacks)
        avg_score, std_return                          = agent.evaluate(env)

        display_and_save_results(agent, self.params, agent.params, self.random_policy_average_return,
                                 self.optimal_score, avg_score, std_return, reward_values, eval_steps,
                                 losses, smooth_sigma=self.plot_smooth_sigma,
                                 agent_params_in_dirname=agent.params.vals_in_dirname)



class TrainingCallback:
    def __init__(self, name):
        self.name = name

    def __call__(self, step, obs=None, action=None, reward=None):
        raise NotImplementedError


class ConvergenceCallback(TrainingCallback, ABC):
    def __init__(self, name):
        super(ConvergenceCallback, self).__init__(name)

    def __call__(self, step, obs=None, action=None, reward=None, **kwargs):
        raise NotImplementedError


class UpperBoundPercentageConvergence(ConvergenceCallback):
    def __init__(self, upper_bound, fraction, n_iters):
        super(UpperBoundPercentageConvergence, self).__init__("Upper bound percentage threshold")
        self.threshold           = upper_bound * fraction
        self.upper_bound         = upper_bound
        self.n_iters             = n_iters
        self.n_consecutive_iters = 0



    def __call__(self, step, obs=None, action=None, reward=None, **kwargs):
        if reward > self.threshold:
            self.n_consecutive_iters += 1
            tqdm.write(f"{step} | " + "Avg reward : "+ colored(str(reward), "green")+f" ({reward/self.upper_bound} of optimal)")
        else:
            self.n_consecutive_iters = 0
            tqdm.write(f"{step} | " + "Avg reward : "+ colored(str(reward), "red")+f" ({reward/self.upper_bound} of optimal)")

        if self.n_consecutive_iters >= self.n_iters:
            return True # "converged"
        else:
            return False




class HistoryCallback(ConvergenceCallback):
    def __init__(self):
        super(HistoryCallback, self).__init__("History")
        self.steps        = []
        self.observations = []
        self.actions      = []
        self.rewards      = []


    def __call__(self, step, obs=None, action=None, reward=None, **kwargs):
        self.steps.append(step)
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

        return False # never converges due to this function









