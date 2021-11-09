import json
import os
from typing import Callable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)


from tensorflow.keras import backend as K

import numpy as np
from tqdm import tqdm

from RL_experiments.environments import RISEnv2
from RL_experiments.standalone_simulatiion import Setup
from RL_experiments.training import NeuralEpsilonGreedyParams


class CustomNeuralEpsilonGreedy2:

    def __init__(self, params: NeuralEpsilonGreedyParams, num_actions:int, observation_dim:int):
        self.name   = "CustomNeuralEpsilonGreedy2"
        self.params = params
        self.num_actions = num_actions
        self.observation_dim = observation_dim
        self.batch_size = self.params.batch_size
        self.params.num_iterations = int(self.params.num_iterations * num_actions)

        obs_inp      = tf.keras.layers.Input((observation_dim,))
        x            = tf.keras.layers.Reshape((-1, 1, 1))(obs_inp)
        x            = tf.keras.layers.Conv2D(64, (5, 1), data_format="channels_last")(x)
        x            = tf.keras.layers.Dropout(self.params.dropout_p)(x)
        x            = tf.keras.layers.MaxPool2D((4, 1))(x)
        x            = tf.keras.layers.Conv2D(64, (5, 1), data_format="channels_last")(x)
        x            = tf.keras.layers.Dropout(self.params.dropout_p)(x)
        x            = tf.keras.layers.MaxPool2D((4, 1))(x)
        x            = tf.keras.layers.Flatten()(x)

        action_input = tf.keras.layers.Input((1,))

        x            = tf.keras.layers.Concatenate()([x, action_input])
        x            = tf.keras.layers.Dense(32, activation='relu')(x)
        x            = tf.keras.layers.Dropout(self.params.dropout_p)(x)
        x            = tf.keras.layers.Dense(32, activation='relu')(x)
        x            = tf.keras.layers.Dropout(self.params.dropout_p)(x)
        out          = tf.keras.layers.Dense(1, activation='linear')(x)

        self.rewardNet = tf.keras.Model([obs_inp, action_input], out)
        optimizer      = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
        self.rewardNet.compile(optimizer, loss='mse')


    def _argmax_action_selection(self, obs):
        obs     = obs.reshape(1,-1)
        rewards = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            a = np.array([i], dtype=np.float32).reshape(1,1)
            rewards[i] = self.rewardNet.predict([obs,a])[0]

        return np.argmax(rewards)

    def _epsilon_greedy_selection(self, obs):
        rnd = np.random.uniform(0, 1)

        if rnd <= self.params.epsilon_greedy:
            return np.random.randint(0, self.num_actions)
        else:
            return self._argmax_action_selection(obs)


    @property
    def policy(self)->Callable:
        return self._argmax_action_selection

    @property
    def collect_policy(self)->Callable:
        return self._epsilon_greedy_selection

    def train(self, env: RISEnv2):

        batch_X   = np.empty((self.batch_size, self.observation_dim))
        batch_a   = np.empty(self.batch_size)
        batch_y   = np.empty(self.batch_size)
        time_step = env._reset()
        batch_i   = 0
        epoch_i   = 0

        rewards      = []
        reward_steps = []
        losses       = []

        initial_reward, _ = self.evaluate(env)

        rewards.append(initial_reward)
        reward_steps.append(0)



        try:
            for step in tqdm(range(self.params.num_iterations)):

                if time_step.is_last():
                    time_step = env._reset()

                obs       = time_step.observation
                action    = self.collect_policy(obs)
                time_step = env._step(action)
                reward    = time_step.reward

                batch_X[batch_i, :] = obs
                batch_y[batch_i]    = reward
                batch_a[batch_i]    = action
                batch_i             = (batch_i + 1) % self.batch_size

                if batch_i % self.batch_size == 0:
                    hist = self.rewardNet.fit([batch_X, batch_a], batch_y, batch_size=self.batch_size, epochs=1, steps_per_epoch=1, verbose=0)
                    epoch_i += 1
                    losses.append(hist.history['loss'][0])

                if (step + 1) % self.params.eval_interval == 0:
                    avg_score, std_score = self.evaluate(env)
                    tqdm.write(f"step={step} | Avg reward = {avg_score} +/- {std_score}.")
                    rewards.append(avg_score)
                    reward_steps.append(step)

        except KeyboardInterrupt:
            print("Training stopped by user...")


        return rewards, losses, reward_steps, self.policy


    def evaluate(self, env: RISEnv2):

        rewards   = np.empty(self.params.num_eval_episodes)
        time_step = env._reset()

        for i in range(self.params.num_eval_episodes):
            if time_step.is_last(): time_step = env._reset()

            obs       = time_step.observation
            action    = self.policy(obs)
            time_step = env._step(action)
            reward    = time_step.reward
            rewards[i] = reward

        return rewards.mean(), rewards.std()





if __name__ == '__main__':
    import sys
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

    env = RISEnv2(setup1, episode_length=np.inf)  # RIS_TFenv(config, 1, transmit_SNR=1)

    print(
        f"\nRunning with {env.action_spec().maximum + 1} actions ({setup1.N_controllable} bits for RIS configurations, {env.codebook_size_bits_required} bits for codebook).")
    print(f"Observation space is of dimension: {env.observation_spec().shape}.")

    train_env = tf_py_environment.TFPyEnvironment(env)
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    optimal_score = compute_average_optimal_policy_return(env, timesteps=agentParams.num_eval_episodes)
    print(f"Score of optimal policy: {optimal_score}\n")

    agent = CustomNeuralEpsilonGreedy2(agentParams, env.action_spec().maximum + 1, env.observation_spec().shape[0])

    # random_policy_average_return = evaluate_agent(random_policy, train_env, num_eval_timesteps, name='Random')
    random_policy_average_return, std_return = compute_avg_return(train_env, random_policy,
                                                                  agentParams.num_eval_episodes)
    print(f"\nRandom policy average return: {random_policy_average_return} +/- {std_return:3f}\n")

    reward_values, losses, eval_steps, best_policy = agent.train(env)

    plot_training_performance(reward_values, eval_steps, agent.name, random_policy_average_return, optimal_score, smooth_sigma=5)

    avg_score, std_return = agent.evaluate(env)


    score_as_percentage_of_random = (avg_score / random_policy_average_return - 1) * 100
    print(f'{agent.name} attained mean performance of {avg_score} +/- {std_return:.3f} ( {score_as_percentage_of_random}% improvement of random policy).')

    score_as_percentage_of_optimal = (avg_score / optimal_score)*100
    print(f'Achieved performance is {score_as_percentage_of_optimal} of average optimal policy.')

    print("Saving results...")

    save_results(agent.name,
                 params['SETUP'],
                 params['NEURAL_EPSILON_GREEDY_PARAMS'],
                 reward_values,
                 eval_steps,
                 {
                     "avg_score"                              : avg_score,
                     "std_return"                             : std_return,
                     "random_policy_average_return"           : random_policy_average_return,
                     "score_as_percentage_of_random"          : score_as_percentage_of_random,
                     "score_as_percentage_of_optimal"         : score_as_percentage_of_optimal
                 },
                 "N_controllable,K,M,codebook_rays_per_RX,kappa_H,observation_noise_variance",
                 "num_iterations,learning_rate"
                 )

    send_notification(f"{agent.name} finished. \n {score_as_percentage_of_random}% above random\n {score_as_percentage_of_optimal}% of optimal")
