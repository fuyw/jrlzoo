"""
mpirun --use-hwthread-cpus python train.py
"""

import os
import sys
import time

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

import numpy as np
import pandas as pd
import torch
from ml_collections import ConfigDict
from mpi4py import MPI

from rl_modules.ddpg_agent import ddpg_agent


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="FetchPush-v3")
    parser.add_argument("--n_epochs", default=50)
    parser.add_argument("--n_cycles", default=50)
    parser.add_argument("--n_batches", default=40)
    parser.add_argument("--save_interval", default=5)
    parser.add_argument("--seed", default=123)
    parser.add_argument("--num_workers", default=1)
    parser.add_argument("--replay_strategy", default="future")
    parser.add_argument("--clip_return", default=50)
    parser.add_argument("--save_dir", default="saved_models")
    parser.add_argument("--noise_eps", default=0.2)
    parser.add_argument("--random_eps", default=0.3)
    parser.add_argument("--buffer_size", default=int(1e6))
    parser.add_argument("--replay_k", default=4)
    parser.add_argument("--clip_obs", default=200)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--gamma", default=0.98)
    parser.add_argument("--max_episode_steps", default=100)
    parser.add_argument("--action_l2", default=1)
    parser.add_argument("--lr_actor", default=0.001)
    parser.add_argument("--lr_critic", default=0.001)
    parser.add_argument("--polyak", default=0.95)
    parser.add_argument("--n_test_rollouts", default=10)
    parser.add_argument("--clip_range", default=5)
    parser.add_argument("--demo_length", default=20)
    parser.add_argument("--cuda", default=False)
    parser.add_argument("--num_rollouts_per_mpi", default=2)
    args = parser.parse_args()
    return args


def get_env_params(env):
    obs, _ = env.reset() 
    params = { 
        "obs": obs["observation"].shape[0],
        "goal": obs["desired_goal"].shape[0],
        "action": env.action_space.shape[0],
        "max_action": env.action_space.high[0],
    }
    params["max_timesteps"] = env._max_episode_steps
    return params


def run(args):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # MPI rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_seed = args.seed + rank

    # initialize the env
    env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps)
    env_params = get_env_params(env)

    # set random seed
    env.action_space.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if args.cuda:
        torch.cuda.manual_seed(rank_seed)

    # initialize the agent
    ddpg_trainer = ddpg_agent(args, env, env_params)

    # start training
    log_info = ddpg_trainer.learn()

    # save log
    if rank == 0:
        df = pd.DataFrame(log_info, columns=["time", "success_rate"])
        df["time"] = (df["time"] - start_time) / 60
        os.makedirs(f"logs/{args.env_name}", exist_ok=True)
        df.to_csv(f"logs/{args.env_name}/s{args.seed}_{timestamp}.csv")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["IN_MPI"] = "1"

    args = get_args()
    run(args)
