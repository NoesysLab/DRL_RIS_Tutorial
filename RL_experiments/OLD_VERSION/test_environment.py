


import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import random_tf_policy
from tf_agents.environments.tf_py_environment import TFPyEnvironment


class TestEnv(py_environment.PyEnvironment):
    def __init__(self, actions=10, noise_variance=0.1, episode_length=10):
        super(TestEnv, self).__init__()

        """
        A contextual bandit environment that at each timestep:
         - samples a number `n` from [0, num_actions-1].
         - emits its binary encoding as observation
         - accepts an action `a`.
         - returns as reward the -|n-a| + epsilon
         - epsilon ~ N(0, noise_variance)
         - Optimal action: encode `n` from binary -> select `n` as action.
        """

        self.episode_length = episode_length
        self.num_actions    = actions
        self.noise_variance = noise_variance
        self._state         = None
        self._t             = None
        self._episode_ended = False
        self.observation_dim = int(np.ceil(np.log2(self.num_actions)))

        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int32,
                                                        minimum=0,
                                                        maximum=self.num_actions - 1,
                                                        name='action')

        self._observation_spec = array_spec.ArraySpec(shape=(self.observation_dim,),
                                                      dtype=np.float32,
                                                      name='observation')


    def _to_binary_array(self, x: int):
        bits = [int(x) for x in bin(x)[2:]]
        rem_padding = self.observation_dim - len(bits)
        zeros = np.zeros((rem_padding,), dtype=int)
        b = np.concatenate([zeros, bits])
        return b

    def _state2observation(self):
        b = self._to_binary_array(self._state)
        return b.astype(np.float32)

    def seed(self, seed_=None):
        pass

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._t = 0
        self._episode_ended = False
        self._state = np.random.randint(low=0, high=self.num_actions)
        return ts.restart(self._state2observation())

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._t += 1

        reward  = -np.abs(action - self._state)
        reward += np.random.normal(0, self.noise_variance)

        #print(f"State: {self._state} | Obs: {self._state2observation()} | Chose: {action} | Reward: {reward}")

        observation = self._state2observation()
        self._state = np.random.randint(low=0, high=self.num_actions)



        if self.episode_length is not None and self._t >= self.episode_length:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward, discount=1.0)

