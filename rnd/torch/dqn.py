import copy
import gym
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hid_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(obs_dim, hid_dim), nn.ReLU()]
        for _ in range(num_layers-1):
            layers.extend([nn.Linear(hid_dim, hid_dim), nn.ReLU()])
        layers.append(nn.Linear(hid_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        q = self.net(x)
        return q


class DQNAgent:
    def __init__(self,
                 lr: float = 1e-3,
                 act_dim: int = 2,
                 obs_dim: int = 2,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 num_layers: int = 2,
                 hid_dim: int = 64):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.tau = tau
        self.qnet = QNetwork(obs_dim, act_dim, hid_dim, num_layers).to(device)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.update_step = 0

    def select_action(self, obs):
        Qs = self.qnet(torch.FloatTensor(obs))
        action = Qs.argmax().item()
        return action

    def loss_fn(self, batch):
        observations, actions, rewards, discounts, next_observations = batch
        Qs = self.qnet(observations)
        Q = torch.gather(Qs, 1, actions).squeeze()
        with torch.no_grad():
            next_Q = self.target_qnet(next_observations).max(1)[0]
        target_Q = rewards + self.gamma * discounts * next_Q
        loss = F.mse_loss(Q, target_Q)
        return loss

    def train(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size)
        self.optimizer.zero_grad()
        loss = self.loss_fn(batch)
        loss.backward()
        self.optimizer.step()
        for param, target_param in zip(self.qnet.parameters(), self.target_qnet.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        return loss


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


# hd: 64-500, 128-225
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
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hid_dim", default=64, type=int)
    parser = parser.parse_args()
    return parser


def run_episode(args):
    t1 = time.time()
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = DQNAgent(lr=args.lr,
                     act_dim=act_dim,
                     obs_dim=obs_dim,
                     hid_dim=args.hid_dim,
                     num_layers=args.num_layers)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(1e5))

    # Warmup
    obs, done = env.reset(), False
    for t in range(args.start_timesteps):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, next_obs, reward, done)
        obs = next_obs
        if done:
            obs, done = env.reset(), False

    # Initialize training stats
    res = [(0, eval_policy(agent, args.env_name, args.seed))]
    for i in range(args.num_episodes):
        obs, done = env.reset(), False
        episode_reward = 0
        episode_step = 0
        while not done:
            if np.random.random() <= args.epsilon:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.add(obs, action, next_obs, reward, done)
            obs = next_obs
            episode_reward += reward
            episode_step += 1
            loss = agent.train(replay_buffer, args.batch_size)
        eval_reward = eval_policy(agent, args.env_name, args.seed)
        if (i+1) % 10 == 0:
            print(f"Episode {i+1}:\teval_reward = {eval_reward:.2f}\ttime = {(time.time()-t1)/60:.2f}\t"
                  f"loss = {loss.item():.2f}\tepisode_step = {episode_step:.0f}\t"
                  f"episode_reward = {episode_reward:.2f}")
            res.append((i+1, eval_reward))
    log_df = pd.DataFrame(res, columns=["step", "reward"])
    avg_reward = log_df["reward"].iloc[-10:].mean()
    return avg_reward, log_df


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    args = get_args()
    for seed in range(3):
        args.seed = seed
        avg_reward, log_df = run_episode(args)
        log_df.to_csv(f"logs/dqn_{args.env_name}_s{args.seed}.csv")
