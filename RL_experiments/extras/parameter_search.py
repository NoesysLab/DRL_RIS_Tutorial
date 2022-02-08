import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf




import json
from copy import copy
from dataclasses import asdict

import numpy as np
from bayes_opt import BayesianOptimization
#from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


import matplotlib.pyplot as plt

from RL_experiments.environments import RISEnv2
from RL_experiments.standalone_simulatiion import Setup
from RL_experiments.train_DQN import DQNAgent, DQNParams
from RL_experiments.train_UCB import UCBAgent, UCBParams
from RL_experiments.training_utils import run_experiment, save_results, plot_loss


def agent_factory(agent_name):
    if agent_name not in ['UCB', 'DQN', 'NeuralEpsilonGreedy']: raise ValueError

    if agent_name == 'UCB':
        return UCBAgent, UCBParams, "UCB_PARAMS"

    elif agent_name == 'DQN':
        return DQNAgent, DQNParams, "DQN_PARAMS"

    elif agent_name == 'NeuralEpsilonGreedy':
        return CustomNeuralEpsilonGreedy, NeuralEpsilonGreedyParams, "NEURAL_EPSILON_GREEDY_PARAMS"

    else:
        assert False



def update_dataclass(obj, dict_values):
    for key, value in dict_values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            raise ValueError



class OptimizationWrapper:

    def __init__(self, agent_name, pbounds, reruns_per_iter=1, params_filename="./parameters.json"):

        self.agent_name  = agent_name
        self.agent_class, agent_params_class, self.agent_params_JSON_key = agent_factory(agent_name)
        self.reruns_per_iter = reruns_per_iter

        general_params    = json.loads(open(params_filename).read())
        agent_params_JSON = general_params[self.agent_params_JSON_key]

        for monitored_param_key in pbounds.keys():
            if monitored_param_key not in agent_params_JSON.keys():
                raise ValueError

        self.agentParams = agent_params_class(**agent_params_JSON)
        self.params_filename = params_filename






    def objective_function(self, **kwargs):
        kwargs["initial_collect_steps"] = int(kwargs["initial_collect_steps"])
        kwargs["target_update_period"]  = int(kwargs["target_update_period"])






        #monitored_params_names = ','.join(kwargs.keys())

        with open(self.params_filename) as f:
            params = json.loads(f.read())
        scores = np.empty(self.reruns_per_iter)

        for run in range(self.reruns_per_iter):
            curr_agent_params = copy(self.agentParams)
            update_dataclass(curr_agent_params, kwargs)
            curr_agent_params.verbose = False

            setup1 = Setup(**params['SETUP'])
            env    = RISEnv2(setup1, episode_length=np.inf)
            agent  = self.agent_class(curr_agent_params,
                                env.action_spec().maximum + 1,
                                env.observation_spec().shape[0])

            reward_values, \
            losses, \
            eval_steps, \
            best_policy = agent.train(env)

            avg_score, \
            std_return = agent.evaluate(env)

            if run == 0: plot_loss(losses, self.agent_name, smooth_sigma=5)

            scores[run] = avg_score

            del setup1, env, agent

        return scores.mean()




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)


    agent = 'DQN'

    pbounds = {
        "dropout_p"                         : (0.01, 0.5),
        "initial_collect_steps"             : (10,   5000),
        "learning_rate"                     : (1e-6, 1e-3),
        "epsilon_greedy"                    : (0.01, 0.5),
        "gradient_clipping"                 : (1e-3, 1e+3),
        "target_update_tau"                 : (1e-7, 1),
        "target_update_period"              : (1,    500),
        "gamma"                             : (0.9,  1),
    }




    wrapper   = OptimizationWrapper(agent, pbounds, reruns_per_iter=3)
    optimizer = BayesianOptimization(f=wrapper.objective_function,
                                     pbounds=pbounds,
                                     random_state=42)

    #logger = JSONLogger(path=f"./results/bayes_opt/{agent}/logs.json")
    #optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    res = optimizer.maximize(init_points=3,
                             n_iter=3)

    print("Bayesian Optimization results:")
    print(json.dumps(optimizer.max['params'], indent=4))
    print(f"Best reward: {optimizer.max['target']}")

