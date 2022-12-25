import gym
import os
import time
import torch
import numpy as np
import pandas as pd

from models import DQNAgent, CQLAgent
from utils import ReplayBuffer, eval_policy, register_custom_envs

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
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--cql_alpha", type=float, default=3.0)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_timesteps", type=int, default=100_000)
    parser.add_argument("--start_timesteps", type=int, default=5_000)
    parser.add_argument("--eval_freq", type=int, default=2_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--plot_traj", action="store_true", default=False)
    parser.add_argument("--epsilon", type=float, default=0.2)
    args = parser.parse_args()
    return args


#################
# Main Function #
#################
def train_and_evaluate(args):
    t1 = time.time()
    exp_name = f"online_{args.agent}_a{args.cql_alpha}_e{args.epsilon}"

    # register pointmass environments
    register_custom_envs()

    # initialize environments
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.n

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize the agent
    agent = AGENT_DICTS[args.agent](lr=args.lr,
                                    obs_dim=obs_dim,
                                    act_dim=act_dim,
                                    hid_dim=args.hid_dim,
                                    cql_alpha=args.cql_alpha)
    agent.load("saved_models/cql/cql_s42")
    reward = eval_policy(agent, eval_env, args.seed)
    print(f"reward for ckpt = {reward:.0f}")

    # create replay buffer
    replay_buffer = ReplayBuffer(obs_dim,
                                 act_dim,
                                 max_size=int(1e5))
    replay_buffer.load("buffers/pointmass.npz")

    # start training
    episode_steps = 0
    obs, done = env.reset(), False
    logs = [{"step":0, "reward":eval_policy(agent, eval_env, args.seed)}]
    for t in range(1, args.max_timesteps+1):
        episode_steps += 1
        if np.random.random() <= args.epsilon:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, done, info = env.step(action)
        done_bool = float(done) if episode_steps < env.unwrapped._max_episode_steps else 0
        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs

        if t > args.start_timesteps: 
            batch = replay_buffer.sample(args.batch_size)
            log_info = agent.update(batch)
            if t % args.eval_freq == 0:
                eval_reward = eval_policy(agent, eval_env, args.seed)
                if args.plot_traj:
                    eval_env.plot_trajectory(f"imgs/online_{args.agent}/{t//args.eval_freq}")
                if args.agent == "cql":
                    print(f"[Step {t}]\n\teval_reward={eval_reward:.2f}\t"
                            f"time={(time.time()-t1)/60:.2f}\t"
                            f"ptr={replay_buffer.ptr}\n\t"
                            f"loss={log_info['avg_loss'].item():.2f}\t"
                            f"mse_loss={log_info['avg_mse_loss'].item():.2f}\t"
                            f"cql_loss={log_info['avg_cql_loss'].item():.2f}\n\t"
                            f"avg_ood_Q={log_info['avg_ood_Q']:.2f}\t"
                            f"avg_Q={log_info['avg_Q']:.2f}\t"
                            f"avg_target_Q={log_info['avg_target_Q']:.2f}\n\n")
                    logs.append({"step": t,
                                "reward": eval_reward,
                                "time": (time.time()-t1)/60,
                                "loss": log_info['avg_loss'].item(),
                                "mse_loss": log_info['avg_mse_loss'].item(),
                                "cql_loss": log_info['avg_cql_loss'].item(),
                                "avg_ood_Q": log_info['avg_ood_Q'].item(),
                                "avg_Q": log_info['avg_Q'].item(),
                                "avg_target_Q": log_info['avg_target_Q'].item()})
                else:
                    print(f"[Step {t}] eval_reward = {eval_reward:.2f}\t"
                            f"time = {(time.time()-t1)/60:.2f}\n\t"
                            f"loss = {log_info['avg_loss'].item():.2f}\t"
                            f"avg_Q = {log_info['avg_Q']:.2f}\t"
                            f"avg_target_Q = {log_info['avg_target_Q']:.2f}\n\n")
                    logs.append({"step": t,
                                "reward": eval_reward,
                                "time": (time.time()-t1)/60,
                                "loss": log_info['avg_loss'].item(),
                                "avg_Q": log_info['avg_Q'].item(),
                                "avg_target_Q": log_info['avg_target_Q'].item()})

    log_df = pd.DataFrame(logs) 
    log_df.to_csv(f"logs/online_{args.agent}/{exp_name}.csv")
    agent.save(f"saved_models/online_{args.agent}/{exp_name}")


if __name__ == "__main__":
    args = get_args()
    os.makedirs(f"logs/online_{args.agent}", exist_ok=True)
    os.makedirs(f"imgs/online_{args.agent}", exist_ok=True)
    os.makedirs(f"saved_models/online_{args.agent}", exist_ok=True)
    train_and_evaluate(args)
