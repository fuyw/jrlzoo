"""
    num_layers  hidden_dim  avg_reward
0            0          32      500.00
1            0          64      494.06
2            0         128      468.96
3            0         256      500.00
4            1          32      367.48
5            1          64      405.81
6            1         128      492.90
7            1         256      488.65
8            2          32      113.23
9            2          64      479.86
10           2         128      493.50
11           2         256      360.48
"""
import gym
import copy
import time
import collections
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(5e4)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, 1), dtype=np.int32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.float32)

    def add(self, observation: np.ndarray, action: np.ndarray,
            next_observation: np.ndarray, reward: float, done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=torch.Tensor(self.observations[idx]),
                      actions=torch.LongTensor(self.actions[idx]),
                      rewards=torch.Tensor(self.rewards[idx]),
                      discounts=torch.Tensor(self.discounts[idx]),
                      next_observations=torch.Tensor(self.next_observations[idx]))
        return batch


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128, num_layers=0):
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
                 num_layers: int = 2,
                 hidden_dim: int = 256):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.tau = tau
        self.qnet = QNetwork(obs_dim, act_dim, hidden_dim, num_layers)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.update_step = 0

    def select_action(self, obs):
        Qs = self.qnet(torch.FloatTensor(obs))
        action = Qs.argmax().item()
        return action

    def loss_fn(self, batch):
        observations, actions, rewards, discounts, next_observations = batch
        Qs = self.qnet(observations)  # (256, 2)
        Q = torch.gather(Qs, 1, actions).squeeze()  # (256,)
        with torch.no_grad():
            next_Q = self.target_qnet(next_observations).max(1)[0]  # (256,)
        target_Q = rewards + self.gamma * discounts * next_Q
        loss = (Q - target_Q)**2
        return loss.mean()

    def train(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size)
        self.optimizer.zero_grad()
        loss = self.loss_fn(batch)
        loss.backward()
        self.optimizer.step()
        for param, target_param in zip(self.qnet.parameters(), self.target_qnet.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return loss


class Agent_notarget:
    def __init__(self,
                 lr: float = 1e-3,
                 act_dim: int = 2,
                 obs_dim: int = 2,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 num_layers: int = 2,
                 hidden_dim: int = 256):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.tau = tau
        self.qnet = QNetwork(obs_dim, act_dim, hidden_dim, num_layers)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.update_step = 0

    def select_action(self, obs):
        Qs = self.qnet(torch.FloatTensor(obs))
        action = Qs.argmax().item()
        return action

    def loss_fn(self, batch):
        observations, actions, rewards, discounts, next_observations = batch
        Qs = self.qnet(observations)  # (256, 2)
        Q = torch.gather(Qs, 1, actions).squeeze()  # (256,)
        with torch.no_grad():
            next_Q = self.qnet(next_observations).max(1)[0]  # (256,)
        target_Q = rewards + self.gamma * discounts * next_Q
        loss = (Q - target_Q)**2
        return loss.mean()

    def train(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size)
        self.optimizer.zero_grad()
        loss = self.loss_fn(batch)
        loss.backward()
        self.optimizer.step()
        return loss


def eval_policy(agent: Agent,
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
    parser.add_argument("--num_layers", default=0, type=int)
    parser.add_argument("--start_timesteps", default=25000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--no_target", default=False, action="store_true")
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

    if args.no_target:
        agent = Agent_notarget(lr=args.lr,
                               act_dim=act_dim,
                               obs_dim=obs_dim,
                               num_layers=args.num_layers,
                               hidden_dim=args.hidden_dim)
    else:
        agent = Agent(lr=args.lr,
                      act_dim=act_dim,
                      obs_dim=obs_dim,
                      num_layers=args.num_layers,
                      hidden_dim=args.hidden_dim)
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
            print(f'Episode {i+1}:\teval_reward = {eval_reward:.2f}\ttime = {(time.time()-t1)/60:.2f}\t'
                  f'loss = {loss.item():.2f}\tepisode_step = {episode_step:.0f}\t'
                  f'episode_reward = {episode_reward:.2f}')
            res.append((i+1, eval_reward))
    log_df = pd.DataFrame(res, columns=['step', 'reward'])
    avg_reward = log_df['reward'].iloc[-10:].mean()
    return avg_reward, log_df


if __name__ == "__main__":
    args = get_args()
    for seed in range(3):
        args.seed = seed
        avg_reward, log_df = run_episode(args)
        log_df.to_csv(f'new_logs/dqn{int(args.no_target)+1}_{args.env_name}_s{args.seed}.csv')
