import gym
import os
import time
import torch
import numpy as np

from models import DQNAgent
from utils import ReplayBuffer

import bsuite
from bsuite.utils import gym_wrapper


###################
# Utils Functions #
###################
AGENT_DICTS = {"dqn": DQNAgent}


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="dqn")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_timesteps", type=int, default=3_000)
    parser.add_argument("--eval_freq", type=int, default=200)
    parser.add_argument("--start_timesteps", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    args = parser.parse_args()
    return args


def eval_policy(agent: DQNAgent,
                eval_env: gym.Env,
                eval_episodes: int = 10) -> float:
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            action = agent.select_action(obs.flatten())  # 2D -> 1D
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


#################
# Main Function #
#################
def train_and_evaluate(args):
    t1 = time.time()

    # initialize bsuite environments
    env = bsuite.load_from_id("catch/0")
    env = gym_wrapper.GymFromDMEnv(env)

    eval_env = bsuite.load_from_id("catch/0")
    eval_env = gym_wrapper.GymFromDMEnv(eval_env)

    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.n

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize the agent
    agent = AGENT_DICTS[args.agent](lr=args.lr, obs_dim=obs_dim, act_dim=act_dim, hid_dim=args.hid_dim)

    # create replay buffer
    replay_buffer = ReplayBuffer(obs_dim,
                                 act_dim,
                                 max_size=int(1e5))

    # start training
    obs, done = env.reset(), False
    logs = [{"step":0, "reward":eval_policy(agent, eval_env, args.seed)}]
    for t in range(1, args.max_timesteps+1):
        if t <= args.start_timesteps: 
            action = env.action_space.sample()
        else:
            if np.random.random() <= args.epsilon:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs.flatten())
        next_obs, reward, done, info = env.step(action)
        done_bool = float(done)
        replay_buffer.add(obs.flatten(), action, next_obs.flatten(), reward, done_bool)
        obs = next_obs

        if t > args.start_timesteps:
            batch = replay_buffer.sample(args.batch_size)
            log_info = agent.update(batch)
            if t % args.eval_freq == 0:
                eval_reward = eval_policy(agent, eval_env, args.seed)
                print(f"[Step {t}] eval_reward = {eval_reward:.2f}\t"
                      f"time = {(time.time()-t1)/60:.2f}\t"
                      f"loss = {log_info['avg_loss'].item():.2f}\t"
                      f"avg_Q = {log_info['avg_Q']:.2f}\t"
                      f"avg_target_Q = {log_info['avg_target_Q']:.2f}")

        if done:
            obs, done = env.reset(), False

    # save the buffer
    replay_buffer.save(f"buffers/catch_{args.agent}")

if __name__ == "__main__":
    os.makedirs("imgs", exist_ok=True)
    os.makedirs("buffers", exist_ok=True)
    args = get_args()
    train_and_evaluate(args)
