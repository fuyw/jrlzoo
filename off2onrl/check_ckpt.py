import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import gym
import d4rl
import time

import numpy as np
from models import CQLAgent
from utils import ReplayBuffer


def get_training_data(replay_buffer, ensemble_num=7, holdout_ratio=0.1, eps=1e-3):
    # load the offline data
    observations = replay_buffer.observations
    actions = replay_buffer.actions
    next_observations = replay_buffer.next_observations
    rewards = replay_buffer.rewards.reshape(-1, 1)  # reshape for concatenate
    holdout_num = int(holdout_ratio * len(observations))

    # validation dataset
    permutation = np.random.permutation(len(observations))
    train_idx, target_idx = permutation[holdout_num:], permutation[:holdout_num]

    # split validation set
    train_observations = observations[train_idx]

    # compute the normalize stats
    obs_mean = train_observations.mean(0)
    obs_std = train_observations.std(0) + eps

    # normlaize the data
    observations = (observations - obs_mean) / obs_std
    next_observations = (next_observations - obs_mean) / obs_std
    delta_observations = next_observations - observations

    # prepare for model inputs & outputs
    inputs = np.concatenate([observations, actions], axis=-1)
    targets = np.concatenate([delta_observations, rewards], axis=-1)

    # split the dataset
    inputs, holdout_inputs = inputs[train_idx], inputs[target_idx]
    targets, holdout_targets = targets[train_idx], targets[target_idx]
    holdout_inputs = np.tile(holdout_inputs[None], [ensemble_num, 1, 1])
    holdout_targets = np.tile(holdout_targets[None], [ensemble_num, 1, 1])

    return inputs, targets, holdout_inputs, holdout_targets, obs_mean, obs_std


env = gym.make("halfcheetah-medium-v2")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
replay_buffer = ReplayBuffer(obs_dim, act_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
inputs, targets, holdout_inputs, holdout_targets, obs_mean, obs_std = get_training_data(replay_buffer)




def eval_policy(agent, eval_env, eval_episodes: int = 10):
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            action = agent.sample_action(obs, True)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


agents = []
scores = []


for cnt in [180, 200]:
    agent = CQLAgent(obs_dim, act_dim)
    agent.load("saved_models/demo", cnt)
    agents.append(agent)
    score, _ = eval_policy(agent, env)
    scores.append(score)

