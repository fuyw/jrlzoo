import os
import gym
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, num_layers=0):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, act_dim)) 
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        q = self.net(x)
        return q


class Agent:
    def __init__(self,
                 lr: float = 1e-3,
                 act_dim: int = 2,
                 obs_dim: int = 2,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 num_layers: int = 0,
                 hidden_dim: int = 256):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.tau = tau
        self.qnet = QNetwork(obs_dim, act_dim, hidden_dim, num_layers)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.update_step = 0

    def select_action(self, obs, epsilon):
        if np.random.random() <= epsilon:
            action = np.random.randint(0, self.act_dim)
        else:
            Qs = self.qnet(torch.FloatTensor(obs))
            action = Qs.argmax().item()
        return action

    def train(self, observation, action, reward, discount, next_observation, next_action):
        self.optimizer.zero_grad()
        Qs = self.qnet(observation)  # (2,)
        Q = torch.gather(Qs, 0, action)  # (1,)
        with torch.no_grad():
            next_Qs = self.qnet(next_observation)
            next_Q = torch.gather(next_Qs, 0, next_action)  # (1,)
        target_Q = reward + self.gamma * discount * next_Q  # (1,)
        loss = (Q - target_Q)**2
        loss.backward()
        self.optimizer.step()
        return loss


def eval_policy(agent: Agent, env_name: str, epsilon: float, seed: int,
                eval_episodes: int = 10) -> float:
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        t = 0
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            action = agent.select_action(obs, epsilon)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="CartPole-v1")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epsilon", default=0.3, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_episodes", default=5000, type=int)
    parser.add_argument("--num_layers", default=0, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
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

    agent = Agent(lr=args.lr,
                  act_dim=act_dim,
                  obs_dim=obs_dim,
                  num_layers=args.num_layers,
                  hidden_dim=args.hidden_dim)

    res = [(0, eval_policy(agent, args.env_name, args.epsilon, args.seed))]
    for i in range(args.num_episodes):
        obs, done = env.reset(), False
        episode_reward = 0
        episode_step = 0
        losses = []
        while not done:
            action = agent.select_action(obs, args.epsilon)
            next_obs, reward, done, _ = env.step(action)
            next_action = agent.select_action(next_obs, args.epsilon)
            loss = agent.train(torch.FloatTensor(obs),
                               torch.LongTensor([action]),
                               reward,
                               1-done,
                               torch.FloatTensor(next_obs),
                               torch.LongTensor([next_action]))
            obs = next_obs
            episode_reward += reward
            episode_step += 1
            losses.append(loss.item())
        eval_reward = eval_policy(agent, args.env_name, args.epsilon, args.seed)
        if (i+1) % (args.num_episodes//100) == 0:
            print(f'Episode {i+1}:\teval_reward = {eval_reward:.2f}\ttime = {(time.time()-t1)/60:.2f}\t'
                  f'loss = {sum(losses)/len(losses):.2f}\tepisode_step = {episode_step:.0f}\t'
                  f'episode_reward = {episode_reward:.2f}')
            res.append((i+1, eval_reward))
    log_df = pd.DataFrame(res, columns=['step', 'reward'])
    avg_reward = log_df['reward'].iloc[-10:].mean(0)
    return avg_reward, log_df


if __name__ == "__main__":
    os.makedirs('new_logs', exist_ok=True)
    args = get_args()
    for epsilon in [0.1, 0.3, 0.5]:
        args.epsilon = epsilon
        for seed in range(3):
            args.seed = seed
            avg_reward, log_df = run_episode(args)
            log_df.to_csv(f'new_logs/sarsa_{args.epsilon}_{args.env_name}_s{args.seed}.csv')