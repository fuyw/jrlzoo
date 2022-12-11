import gym
import os
import time
import torch
import numpy as np

from models import DQNAgent, CQLAgent
from utils import ReplayBuffer, register_custom_envs


###################
# Utils Functions #
###################
AGENT_DICTS = {"cql": CQLAgent, "dqn": DQNAgent}


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="PointmassHard-v0", choices=(
        "PointmassEasy-v0", "PointmassMedium-v0", "PointmassHard-v0", "PointmassVeryHard-v0"))
    parser.add_argument("--agent", default="dqn", choices=("cql", "dqn"))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_timesteps", type=int, default=50_000)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--start_timesteps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
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
            action = agent.select_action(obs)
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
    register_custom_envs()

    # initialize environments
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
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

    # start training
    episode_steps = 0
    obs, done = env.reset(), False
    logs = [{"step":0, "reward":eval_policy(agent, eval_env, args.seed)}]
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
        done_bool = float(done) if episode_steps < env.unwrapped._max_episode_steps else 0
        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs

        if t > args.start_timesteps:
            batch = replay_buffer.sample(args.batch_size)
            log_info = agent.update(batch)
            if t % args.eval_freq == 0:
                eval_reward = eval_policy(agent, eval_env, args.seed)
                if args.plot_traj:
                    eval_env.plot_trajectory(f"imgs/{t//args.eval_freq}")
                print(f"[Step {t}] eval_reward = {eval_reward:.2f}\t"
                      f"time = {(time.time()-t1)/60:.2f}\t"
                      f"loss = {log_info['avg_loss'].item():.2f}\t"
                      f"avg_Q = {log_info['avg_Q']:.2f}\t"
                      f"avg_target_Q = {log_info['avg_target_Q']:.2f}")

        if done:
            obs, done = env.reset(), False
            episode_steps = 0

    # save the buffer
    replay_buffer.save(f"buffers/{args.agent}")

if __name__ == "__main__":
    os.makedirs("imgs", exist_ok=True)
    os.makedirs("buffers", exist_ok=True)
    args = get_args()
    train_and_evaluate(args)
