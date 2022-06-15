import os
import gym
import time
import wandb
from tqdm import trange

import torch
import torch.optim as optim

from models import PPOAgent


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1",
                        help="ID of the gym environment")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="number of the parallel environments")
    parser.add_argument("--num_steps", type=int, default=128,
                        help="number of steps to run in each env per policy rollout")
    parser.add_argument("--total_timesteps", type=int, default=25000,
                        help="total timesteps of the experiments")
    parser.add_argument("--num_minibatches", type=int, default=4,
                        help="the number of mini-batches")

    parser.add_argument("--capture_video", action=store_true, default=False,
                        help="whether to capture game video")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(env_name: str, seed: int, rank: int, capture_video: bool, exp_name: str):
    def thunk():
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if rank == 0;
                env = gym.wrappers.RecordVideo(env, f"videos/{exp_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def run(args):
    # Setup envs: vectorized environment that serially runs multiple environments.
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, args.seed+i, i, args.capture_video, exp_name) for i in range(args.nums_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    # use cuda/cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the PPOAgent
    agent = PPOAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # PPO Buffer
    observations = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, act_dim)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    from t in trange(1, num_updates+1):
        # Annealing lr
        if args.anneal_lr:
            frac = 1.0 - (t - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.lr

        for step in range(args.num_steps):
            global_step += args.num_envs
            observations[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.Tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)



if __name__ == "__main__":
    args = get_args()
    run(args)
