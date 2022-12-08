import gym
import os
import time
import torch
import numpy as np
import pandas as pd

from tqdm import trange

from models import DQNAgent
from utils import ReplayBuffer


###################
# Utils Functions #
###################
def eval_policy(agent: DQNAgent,
                env_name: str,
                seed: int,
                eval_episodes: int = 10) -> float:
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        t = 0
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            action = agent.select_action(obs)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="CartPole-v1")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epsilon", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_episodes", default=1000, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--start_timesteps", default=25000, type=int)
    parser.add_argument("--max_timesteps", default=int(2e5), type=int)
    parser.add_argument("--eval_freq", default=int(1e4), type=int)
    parser.add_argument("--update_target_freq", default=4, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hid_dim", default=64, type=int)
    parser = parser.parse_args()
    return parser


#################
# Main Function #
#################
def train_and_evaluate(args):
    t1 = time.time()
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize the agent
    agent = DQNAgent(lr=args.lr,
                     act_dim=act_dim,
                     obs_dim=obs_dim,
                     hid_dim=args.hid_dim,
                     num_layers=args.num_layers)

    # create replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(1e5))

    # start training
    episode_steps = 0
    obs, done = env.reset(), False
    logs = [{"step":0, "reward":eval_policy(agent, args.env_name, args.seed)}]
    for t in range(1, args.max_timesteps+1):
        episode_steps += 1
        if t <= args.start_timesteps: 
            action = env.action_space.sample()
        else:
            if np.random.random() <= args.epsilon:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

        next_obs, reward, done, info = env.step(action)
        done_bool = float(done) if 'TimeLimit.truncated' not in info else 0
        # done_bool = float(done) if episode_steps < env._max_episode_steps else 0

        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs

        if done:
            obs, done = env.reset(), False
            episode_steps = 0

        if t > args.start_timesteps:
            batch = replay_buffer.sample(args.batch_size)
            log_info = agent.update(batch)
            if t % args.eval_freq == 0:
                eval_reward = eval_policy(agent, args.env_name, args.seed)
                print(f"[Step {t}] eval_reward = {eval_reward:.2f}\t"
                      f"time = {(time.time()-t1)/60:.2f}\t"
                      f"loss = {log_info['avg_loss'].item():.2f}\t"
                      f"avg_Q = {log_info['avg_Q']:.2f}\t"
                      f"avg_target_Q = {log_info['avg_target_Q']:.2f}\t"
                      f"episode_steps = {episode_steps:.0f}")


if __name__ == "__main__":
    args = get_args()
    train_and_evaluate(args)
