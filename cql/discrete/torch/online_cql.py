import gym
import os
import time
import torch
import numpy as np
import pandas as pd

from models import DQNAgent, CQLAgent
from utils import ReplayBuffer, register_custom_envs

import bsuite
from bsuite.utils import gym_wrapper


###################
# Utils Functions #
###################
AGENT_DICTS = {"cql": CQLAgent, "dqn": DQNAgent}


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="PointmassHard-v2")
    parser.add_argument("--agent", default="cql", choices=("cql", "dqn"))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cql_alpha", type=float, default=1.0)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_timesteps", type=int, default=50_000)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--plot_traj", action="store_true", default=False)
    args = parser.parse_args()
    return args


def eval_policy(agent: CQLAgent,
                eval_env: gym.Env,
                eval_episodes: int = 10) -> float:
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            action = agent.select_action(obs.flatten())
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def collect_trajectory(agent: CQLAgent, env: gym.Env, epsilon: float = 0.1):
    obs, done = env.reset(), False
    trajectories = [obs]
    while not done:
        trajectories.append(obs)
        if np.random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs.flatten())
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
    return trajectories


#################
# Main Function #
#################
def train_and_evaluate(args):
    t1 = time.time()

    # register pointmass environments
    register_custom_envs()

    # initialize environments
    env = gym.make(args.env_name)
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.n

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize the agent
    exp_name = f"cql_alpha{args.cql_alpha}"
    agent = AGENT_DICTS[args.agent](lr=args.lr,
                                    obs_dim=obs_dim,
                                    act_dim=act_dim,
                                    hid_dim=args.hid_dim,
                                    cql_alpha=args.cql_alpha)

    # create replay buffer
    replay_buffer = ReplayBuffer(obs_dim,
                                 act_dim,
                                 max_size=int(1e5))
    replay_buffer.load("buffers/pointmass.npz")

    # start training
    obs = env.reset()
    logs = [{"step":0, "reward":eval_policy(agent, env, args.seed)}]
    for t in range(1, args.max_timesteps+1):
        batch = replay_buffer.sample(args.batch_size)
        log_info = agent.update(batch)
        if t % args.eval_freq == 0:
            eval_reward = eval_policy(agent, env, args.seed)
            if args.plot_traj:
                env.plot_trajectory(f"imgs/{args.agent}/{t//args.eval_freq}")
            print(f"[Step {t}] eval_reward = {eval_reward:.2f}\t"
                    f"time = {(time.time()-t1)/60:.2f}\n\t"
                    f"loss = {log_info['avg_loss'].item():.2f}\t"
                    f"mse_loss = {log_info['avg_mse_loss'].item():.2f}\t"
                    f"cql_loss = {log_info['avg_cql_loss'].item():.2f}\n\t"
                    f"avg_ood_Q = {log_info['avg_ood_Q']:.2f}\t"
                    f"avg_Q = {log_info['avg_Q']:.2f}\t"
                    f"avg_target_Q = {log_info['avg_target_Q']:.2f}\n\n")
            logs.append({"step": t,
                         "reward": eval_reward,
                         "time": (time.time()-t1)/60,
                         "loss": log_info['avg_loss'].item(),
                         "mse_loss": log_info['avg_mse_loss'].item(),
                         "cql_loss": log_info['avg_cql_loss'].item(),
                         "avg_ood_Q": log_info['avg_ood_Q'].item(),
                         "avg_Q": log_info['avg_Q'].item(),
                         "avg_target_Q": log_info['avg_target_Q'].item()})
    log_df = pd.DataFrame(logs) 
    log_df.to_csv(f"logs/{args.agent}/{exp_name}.csv")
    agent.save(f"saved_models/{args.agent}/{exp_name}")


if __name__ == "__main__":
    args = get_args()
    os.makedirs(f"logs/{args.agent}", exist_ok=True)
    os.makedirs(f"saved_models/{args.agent}", exist_ok=True)
    os.makedirs(f"imgs/{args.agent}", exist_ok=True)
    train_and_evaluate(args)
