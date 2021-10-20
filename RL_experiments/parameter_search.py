import json
from copy import copy

import numpy as np
from bayes_opt import BayesianOptimization
#from bayes_opt import SequentialDomainReductionTransformer
import matplotlib.pyplot as plt

from RL_experiments.train_DQN import DQNAgent, DQNParams
from RL_experiments.train_Neural_Epsilon_Greedy import NeuralEpsilonGreedyParams, CustomNeuralEpsilonGreedy
from RL_experiments.train_UCB import UCBAgent, UCBParams
from RL_experiments.training_utils import run_experiment


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

    def __init__(self, agent_name, pbounds, params_filename="./parameters.json"):

        self.agent_class, agent_params_class, self.agent_params_JSON_key = agent_factory(agent_name)

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

        curr_agent_params = copy(self.agentParams)
        update_dataclass(curr_agent_params, kwargs)

        monitored_params_names = ','.join(kwargs.keys())

        score, _ = run_experiment(self.params_filename, self.agent_class, None,
                                  self.agent_params_JSON_key, monitored_params_names,
                                  agentParams=curr_agent_params)

        return score




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


    test_values = {
        "dropout_p": 0.2,
        "initial_collect_steps": 100,
        "learning_rate": 1e-5,
        "epsilon_greedy": 0.3,
        "gradient_clipping": 1e-2,
        "target_update_tau": 0.01,
        "target_update_period": 100,
        "gamma": 0.99,
    }




    wrapper   = OptimizationWrapper(agent, pbounds)
    # optimizer = BayesianOptimization(f=wrapper.objective_function,
    #                                  pbounds=pbounds,
    #                                  random_state=32)

    # res = optimizer.maximize(init_points=3,
    #                          n_iter=3)

    wrapper.objective_function(**test_values)