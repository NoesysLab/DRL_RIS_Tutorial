import itertools
import sys
from typing import Any

from tqdm import tqdm

from RL_experiments.standalone_simulatiion import Setup, compute_sum_rate, compute_SINR_per_user, simulate_transmission

sys.path.insert(0,'..')

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts



class RISEnv2(py_environment.PyEnvironment):


    def get_state(self) -> Any:
        raise NotImplemented

    def set_state(self, state: Any) -> None:
        raise NotImplemented

    def __init__(self, setup: Setup, episode_length):
        super(RISEnv2, self).__init__()
        self.setup = setup

        self.episode_length      = episode_length
        self._t                  = None
        self._episode_ended      = False
        self._prev_configuration = None
        self.observation_noise_variance = setup.observation_noise_variance

        self.num_RIS_configurations      = None
        self.codebook_size_bits_required = None

        self._state = None

        action_dim, observation_dim = self._calculate_space_dimensions()

        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int32,
                                                        minimum=0,
                                                        maximum=action_dim - 1,
                                                        name='action')

        self._observation_spec = array_spec.ArraySpec(shape=(observation_dim,),
                                                      dtype=np.float32,
                                                      name='observation')

        ################# K ###################
        self._info = {}


    def _calculate_space_dimensions(self):

        M = self.setup.M
        B = self.setup.B
        K = self.setup.K
        N = self.setup.N
        #N_tot = self.setup.N_tot
        N_controllable = self.setup.N_controllable



        self.num_RIS_configurations      = int(np.power(2, N_controllable))
        self.codebook_size_bits_required = self.setup.K * int(np.ceil(np.log2(self.setup.codebook_rays_per_RX)))
        action_dim                       = int(np.power(2, N_controllable + self.codebook_size_bits_required))


        observation_dim = ((M * B * N) + (M * K * N) + (K * B)) *      2
        #                   +-------+    +---------+   +------+  +---------+
        #                       H             G           h       real+imag

        return action_dim, observation_dim







    def generate_observation_vector(self, H, G, h):
        H   = H.flatten()
        G   = G.flatten()
        h   = h.flatten()
        obs = np.concatenate([H.real, H.imag, G.real, G.imag, h.real, h.imag])
        obs = obs.astype(np.float32).flatten()

        if self.observation_noise_variance != 0:
            obs += np.random.normal(loc=0, scale=self.observation_noise_variance, size=len(obs))

        return obs

    def compute_reward(self, H, G, h, RIS_profiles, W):
        SINR = compute_SINR_per_user(self.setup,
                                     H, G, h,
                                     RIS_profiles,
                                     W)
       # return float(np.sum(SINR))
        return compute_sum_rate(SINR)

    def action_2_configuration(self, a):
        bits = np.array([int(x) for x in list('{0:0b}'.format(a))], dtype=int)
        rem_padding = self.setup.N_controllable - len(bits)
        zeros = np.zeros((rem_padding,), dtype=int)
        configuration = np.concatenate([zeros, bits])
        return configuration


    def extend_configuration_to_groups(self, configuration: np.ndarray):
        assert configuration.ndim == 1
        assert len(configuration) == self.setup.N_controllable

        configuration  = np.repeat(configuration, self.setup.group_size)

        assert len(configuration) == self.setup.N_tot

        return configuration

    def configuration2phases(self, configuration):
        # thetas = configuration*np.pi # convert {0,1} to {0, pi}
        # Phi = np.exp(1j*thetas)
        Phi = 2 * configuration - 1  # exp(j*theta) for theta in {0, pi} results in {-1,1}. So better to do it manually for numerical stability
        Phi = Phi.reshape((self.setup.M, self.setup.N))
        return Phi


    def split_action_to_precoding_and_RIS_profile(self, action):
        bits                   = np.array([int(x) for x in list('{0:0b}'.format(action))], dtype=int)
        rem_padding            = self.codebook_size_bits_required + self.setup.N_controllable - len(bits)
        zeros                  = np.zeros((rem_padding,), dtype=int)
        whole_action_bits      = np.concatenate([zeros, bits])
        precoding_bits         = whole_action_bits[:self.codebook_size_bits_required]
        RIS_configuration_bits = whole_action_bits[self.codebook_size_bits_required:]


        return precoding_bits, RIS_configuration_bits



    def select_beamforming_matrix(self, codebook_selection: np.ndarray):
        # codebook: (4, B, K)
        binary_array_to_int    = lambda b: int(b.dot(2**np.arange(b.size)[::-1]))
        bits_per_RX            = self.codebook_size_bits_required // self.setup.K

        W = np.empty(shape=(self.setup.B, self.setup.K), dtype=complex)
        for k in range(self.setup.K):
            this_user_codebook_bits = codebook_selection[k*bits_per_RX:(k+1)*bits_per_RX]
            i                       = binary_array_to_int(this_user_codebook_bits)
            W[:,k]                  = self.setup.codebook[i, :, k]

        return W


    def seed(self, seed_=None):
        pass

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._t = 0
        self._episode_ended = False
        self._prev_configuration = np.zeros(shape=self._observation_spec.shape, dtype=np.float32)
        H, G, h = simulate_transmission(self.setup)
        self._state = (H, G, h)
        obs = self.generate_observation_vector(H, G, h)

        ################# K ###################
        self._info = {}

        return ts.restart(obs)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        codebook_selection, configuration = self.split_action_to_precoding_and_RIS_profile(action)



        #configuration = self.action_2_configuration(action)
        configuration = self.extend_configuration_to_groups(configuration)
        Phi           = self.configuration2phases(configuration)
        W             = self.select_beamforming_matrix(codebook_selection)
        H, G, h       = self._state
        reward        = self.compute_reward(H, G, h, Phi, W)
        H, G, h       = simulate_transmission(self.setup)
        observation   = self.generate_observation_vector(H, G, h)

        self._state   = (H, G, h)
        self._t      += 1
        ################# K ###################
        self._info    = {'t': self._t-1, 'configuration': configuration, 'W': W}


        if self.episode_length is not None and self._t >= self.episode_length:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward, discount=1.0)


    def get_info(self) -> Any:
        ################# K ###################
        return self._info


# -------------------------------------------------------------




################# K ###################
class RateRequestsEnv(RISEnv2):

    def compute_reward(self, H, G, h, RIS_profiles, W):
        SINR = compute_SINR_per_user(self.setup,
                                     H, G, h,
                                     RIS_profiles,
                                     W)
        requests_penalty = np.clip(SINR-self.setup.rate_requests, a_min=None, a_max=0).sum()
        self._info['SINR'] = SINR
        return requests_penalty



################# K ###################
class RateQoSEnv(RateRequestsEnv):

    def compute_reward(self, H, G, h, RIS_profiles, W):
        requests_penalty = super().compute_reward(H, G, h, RIS_profiles, W)

        SINR = self._info['SINR']
        self._info['requests_penalties'] = np.clip(SINR-self.setup.rate_requests, a_min=None, a_max=0)

        if requests_penalty >= 0:
            return compute_sum_rate(SINR)
        else:
            return requests_penalty




# ----------------------------------------------------------------








def compute_average_optimal_policy_return(env: RISEnv2, timesteps=1000):

    total_return = 0
    _ = env._reset()


    for _ in tqdm(range(timesteps)):
        r_max  = -float('inf')
        a_best = -1
        (H, G, h) = env._state
        for a in range(env.action_spec().maximum):
            codebook_selection, configuration = env.split_action_to_precoding_and_RIS_profile(a)
            configuration                     = env.extend_configuration_to_groups(configuration)
            Phi                               = env.configuration2phases(configuration)
            W                                 = env.select_beamforming_matrix(codebook_selection)
            r                                 = env.compute_reward(H, G, h, Phi, W)
            if r > r_max:
                r_max, a_best = r, a

        total_return += r_max
        _ = env._step(a_best)

    return total_return / timesteps






















# -------------------------------------------------------------------
class BanditWrapper:
    def __init__(self, tf_env):
        self.env = tf_env

    def generate_reward(self, action):
        time_step = self.env._step(action)
        return time_step.reward.numpy()[0]








if __name__ == '__main__':
    setup = Setup.load_from_json("./parameters.json")
    env = RISEnv2(setup, 1)

    score = compute_average_optimal_policy_return(env, 1000)
    print(f"Optimal policy avg return: {score}")