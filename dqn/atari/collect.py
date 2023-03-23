import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import time

import gym
import numpy as np
import pandas as pd
from tqdm import trange, tqdm

from atari_wrappers import wrap_deepmind
from models import DQNAgent
from utils import Experience, ReplayBuffer, get_logger, linear_schedule


###################
# Utils Functions #
###################
def eval_policy(agent, env):
    t1 = time.time()
    obs = env.reset()
    act_counts = np.zeros(env.action_space.n)
    while not env.get_real_done():
        # action = agent.sample(np.moveaxis(obs[None], 1, -1))
        action = agent.sample(obs[None])
        act_counts[action] += 1
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    act_counts /= act_counts.sum()
    act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
    return np.mean(env.get_eval_rewards()), act_counts, time.time() - t1


#################
# Main Function #
#################
def train_and_evaluate(config):
    # general setting
    start_time = time.time() 

    # make environments
    env = gym.make(f"{config.env_name}NoFrameskip-v4")
    env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")
    act_dim = env.action_space.n

    # initialize DQN agent
    agent = DQNAgent(act_dim=act_dim,
                     seed=config.seed,
                     lr_start=config.lr_start,
                     lr_end=config.lr_end,
                     total_timesteps=config.total_timesteps//config.train_freq)

    # create the replay buffer
    replay_buffer = ReplayBuffer(max_size=int(2.5e6))
    L = 200_000

    ckpt_dir = os.listdir(f"saved_models/{config.env_name}")[0]
    for i in range(1, 11):
        agent.load(f"saved_models/{config.env_name}/{ckpt_dir}", 10)
        cnts = 0
        obs = env.reset()
        for j in trange(L):
            if np.random.random() < 0.2:
                action = np.random.choice(act_dim)
            else:
                context = replay_buffer.recent_obs()
                context.append(obs)
                context = np.stack(context, axis=-1)[None]
                action = agent.sample(context)

            # interact with the environment
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.add(Experience(obs, action, reward, done))
            obs = next_obs
            if done:
                obs = env.reset()
    replay_buffer.save(f"{config.env_name}")
    