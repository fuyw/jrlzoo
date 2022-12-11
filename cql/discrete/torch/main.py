import gym
import os
import time
import torch
import numpy as np

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
    parser.add_argument("--env_name", default="PointmassHard-v0", choices=(
        "PointmassEasy-v0", "PointmassMedium-v0", "PointmassHard-v0", "PointmassVeryHard-v0"))
    parser.add_argument("--agent", default="cql", choices=("cql", "dqn"))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--max_timesteps", type=int, default=50_000)
    # parser.add_argument("--eval_freq", type=int, default=1000)
    # parser.add_argument("--start_timesteps", type=int, default=2000)
    parser.add_argument("--max_timesteps", type=int, default=3_000)
    parser.add_argument("--eval_freq", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--plot_traj", action="store_true", default=True)
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
            action = agent.select_action(obs.flatten())
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


#################
# Main Function #
#################
def train_and_evaluate(args):
    t1 = time.time()

    # register pointmass environments
    # register_custom_envs()

    # initialize environments
    # env = gym.make(args.env_name) 

    env = bsuite.load_and_record_to_csv('catch/0', results_dir='./')
    env = gym_wrapper.GymFromDMEnv(env)

    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.n

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize the agent
    agent = AGENT_DICTS[args.agent](lr=args.lr,
                                    obs_dim=obs_dim,
                                    act_dim=act_dim,
                                    hid_dim=args.hid_dim)

    # create replay buffer
    replay_buffer = ReplayBuffer(obs_dim,
                                 act_dim,
                                 max_size=int(1e5))
    replay_buffer.load("buffers/catch_dqn.npz")

    # start training
    obs, done = env.reset(), False
    logs = [{"step":0, "reward":eval_policy(agent, env, args.seed)}]
    for t in range(1, args.max_timesteps+1): 
        batch = replay_buffer.sample(args.batch_size)
        log_info = agent.update(batch)
        if t % args.eval_freq == 0:
            eval_reward = eval_policy(agent, env, args.seed)
            # if args.plot_traj:
            #     env.plot_trajectory(f"imgs/{args.agent}/{t//args.eval_freq}")
            print(f"[Step {t}] eval_reward = {eval_reward:.2f}\t"
                    f"time = {(time.time()-t1)/60:.2f}\t"
                    f"loss = {log_info['avg_loss'].item():.2f}\t"
                    f"mse_loss = {log_info['avg_mse_loss'].item():.2f}\t"
                    f"cql_loss = {log_info['avg_cql_loss'].item():.2f}\t"
                    f"avg_Q = {log_info['avg_Q']:.2f}\t"
                    f"avg_ood_Q = {log_info['avg_ood_Q']:.2f}\t"
                    f"avg_target_Q = {log_info['avg_target_Q']:.2f}\t"
                    )


if __name__ == "__main__":
    args = get_args()
    os.makedirs(f"imgs/{args.agent}", exist_ok=True)
    train_and_evaluate(args)
