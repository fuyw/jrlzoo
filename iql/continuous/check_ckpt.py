import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import gym
import d4rl
import time

import numpy as np
from models import IQLAgent
from utils import ReplayBuffer


def eval_policy(agent, eval_env, eval_episodes: int = 10):
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            action = agent.sample_action(obs)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


agents = []
scores = []
env = gym.make("halfcheetah-medium-v2")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]


fnames = os.listdir("saved_models/halfcheetah-medium-v2")
for fname in fnames:
    agent = IQLAgent(obs_dim, act_dim)
    agent.load(f"saved_models/halfcheetah-medium-v2/{fname}", 200)
    agents.append(agent)
    score, _ = eval_policy(agent, env)
    scores.append(score)

