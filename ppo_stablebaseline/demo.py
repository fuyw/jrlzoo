import os
import gym
import base64
from pathlib import Path
import numpy as np

from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


def evaluate(model, num_episodes=100):
    """
    Evaluate an RL agent
    """
    # Only work for a single environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)

            # here, action, rewards, and dones are arrays
            # because we are using vectorized env (1 proc)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
        all_episode_rewards.append(sum(episode_rewards))
    mean_episode_reward = np.mean(all_episode_rewards)
    print(f'Mean reward: {mean_episode_reward:.2f}, Num episodes: {num_episodes}')
    return mean_episode_reward


def run():
    # create an environment
    env = gym.make('CartPole-v1')

    # initialize the model
    model = PPO('MlpPolicy', env, verbose=1)

    # evaluate the untrained model
    mean_reward_before_train = evaluate(model, num_episodes=100)

    # train the model for 10000 steps
    model.learn(total_timesteps=10_000)

    # evaluate the trained model
    mean_reward_after_train = evaluate(model, num_episodes=100)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()

