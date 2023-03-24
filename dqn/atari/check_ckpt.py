import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import time

import gym
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from atari_wrappers import wrap_deepmind
from models import DQNAgent, CQLAgent
from utils import Experience


###################
# Utils Functions #
###################
def eval_policy(agent, env):
    t1 = time.time()
    obs = env.reset()
    act_counts = np.zeros(env.action_space.n)
    with tqdm(total=int(1e5)) as pbar:
        while not env.get_real_done():
            action = agent.sample(obs[None])
            act_counts[action] += 1
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
            pbar.update(1)
    act_counts /= act_counts.sum()
    act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
    return np.mean(env.get_eval_rewards()), act_counts, time.time() - t1

# Asterix: 4866
# Breakout: 150

env_name = "Asterix"
ckpt_dir = f"saved_models/cql/{env_name}"

env = gym.make(f"{env_name}NoFrameskip-v4")
env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")
eval_env = gym.make(f"{env_name}NoFrameskip-v4")
eval_env = wrap_deepmind(eval_env, dim=84, obs_format="NHWC", test=True)
act_dim = env.action_space.n

for i in range(1, 2):
    # agent = DQNAgent(act_dim=act_dim)
    agent = CQLAgent(act_dim=act_dim)
    agent.load(f"{ckpt_dir}", i)
    eval_reward, act_counts, eval_time = eval_policy(agent, eval_env)
    print(f"{i}: {eval_reward:.3f}, {eval_time:.2f}\n\t{act_counts}")
