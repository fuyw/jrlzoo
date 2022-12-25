import gym
import os
import time
import torch
import numpy as np
import pandas as pd

from models import DQNAgent, CQLAgent
from utils import ReplayBuffer, eval_policy, register_custom_envs


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
    parser.add_argument("--cql_alpha", type=float, default=3.0)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_timesteps", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--plot_traj", action="store_true", default=False)
    args = parser.parse_args()
    return args


#################
# Main Function #
#################
def train_and_evaluate(args):
    t1 = time.time()
    exp_name = f"{args.agent}_s{args.seed}"

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
    logs = [{"step":0, "reward":eval_policy(agent, env, args.seed)}]
    for t in range(1, args.max_timesteps+1):
        batch = replay_buffer.sample(args.batch_size)
        log_info = agent.update(batch)
        if t % args.eval_freq == 0:
            eval_reward = eval_policy(agent, env, args.seed)
            if args.plot_traj:
                env.plot_trajectory(f"imgs/{args.agent}/{t//args.eval_freq}")
            if args.agent == "cql":
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
    log_df.to_csv(f"logs/{args.agent}/{exp_name}.csv")
    agent.save(f"saved_models/{args.agent}/{exp_name}")


if __name__ == "__main__":
    args = get_args()
    os.makedirs(f"logs/{args.agent}", exist_ok=True)
    os.makedirs(f"imgs/{args.agent}", exist_ok=True)
    os.makedirs(f"saved_models/{args.agent}", exist_ok=True)
    train_and_evaluate(args)
