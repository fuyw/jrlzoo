import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import time

import d4rl
import gym
import numpy as np
import pandas as pd
from tqdm import tqdm
from models import TD3BCAgent


def eval_policy(agent,
                eval_env,
                obs_mean: np.ndarray = 0.0,
                obs_std: np.ndarray = 1.0,
                eval_episodes: int = 10):
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            obs = (obs - obs_mean)/obs_std
            action = agent.sample_action(obs)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


mujoco_envs = [
    "halfcheetah-random-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "hopper-random-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "walker2d-random-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
]

antmaze_envs = [
    "antmaze-umaze-v0",
    "antmaze-umaze-diverse-v0",
    "antmaze-medium-play-v0",
    "antmaze-medium-diverse-v0",
    "antmaze-large-play-v0",
    "antmaze-large-diverse-v0",
]


def save_td3bc_stats():
    td3bc_res = {}
    for env_name in mujoco_envs:
        env = gym.make(env_name)
        ds = d4rl.qlearning_dataset(env)
        ds_observations = ds["observations"]
        obs_mean = ds_observations.mean(0)
        obs_std  = ds_observations.std(0) + 1e-3
        td3bc_res[env_name] = {"obs_mean": obs_mean,
                            "obs_std": obs_std}
    np.save("td3bc_obs_stats", td3bc_res)
    data = np.load("td3bc_obs_stats.npy", allow_pickle=True)


res = []
for env_name in tqdm(mujoco_envs):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = TD3BCAgent(obs_dim, act_dim)
    obs_mean = td3bc_res[env_name]["obs_mean"]
    obs_std = td3bc_res[env_name]["obs_std"]
    agent.load(f"saved_ckpts/{env_name}", 200)
    score, _ = eval_policy(agent, env, obs_mean, obs_std)
    res.append((env_name, score))
