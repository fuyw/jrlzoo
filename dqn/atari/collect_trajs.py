import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import time

import gym
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from atari_wrappers import wrap_deepmind
from models import DQNAgent
from utils import Experience, ReplayBuffer, get_logger, linear_schedule


###################
# Utils Functions #
###################
env_name = "Breakout"
ckpt_dir = os.listdir(f"saved_models/{env_name}")[0]


env = gym.make(f"{env_name}NoFrameskip-v4")
env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")
act_dim = env.action_space.n
agent = DQNAgent(act_dim=act_dim)


replay_buffer = ReplayBuffer(max_size=int(2.5e6))


L = 200_000
for i in range(1, 11):
    agent.load(f"saved_models/{env_name}/{ckpt_dir}", i)
    cnts = 0
    with tqdm(total=L) as pbar:
        steps = 0
        while cnts < L: 
            obs = env.reset()
            if np.random.random() < 0.2:
                action = np.random.choice(act_dim)
            else:
                context = replay_buffer.recent_obs()
                context.append(obs)
                context = np.stack(context, axis=-1)[None]
                action = agent.sample(context)
            steps += 1
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.add(Experience(obs, action, reward, done))
            obs = next_obs
            if done:
                obs = env.reset()
                cnts += steps
                pbar.update(steps)
                steps = 0
