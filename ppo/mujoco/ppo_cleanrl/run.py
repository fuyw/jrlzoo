from typing import Tuple
import gym
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import trange
from env_utils import create_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_policy(agent, env, eval_episodes: int = 10) -> Tuple[float]:
    """For-loop sequential evaluation."""
    t1 = time.time()
    avg_reward = 0.
    eval_step = 0
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            with torch.no_grad():
                action, _ = agent.sample_actions(torch.Tensor(obs[None]).to(device))
            next_obs, reward, done, _ = env.step(action.cpu().numpy().squeeze(0))
            avg_reward += reward
            eval_step += 1
            obs = next_obs
    avg_reward /= eval_episodes
    return avg_reward, eval_step, time.time() - t1


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor_mu = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_values(self, observations):
        return self.critic(observations).squeeze(-1)

    def sample_actions(self, observations):
        mu = self.actor_mu(observations)
        log_std = self.actor_logstd.expand_as(mu)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(axis=-1)
        return actions.clip(-0.99999, 0.99999), log_probs

    def get_logp(self, observations, actions):
        mu = self.actor_mu(observations)
        log_std = self.actor_logstd.expand_as(mu)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return log_probs, entropy


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor_num", type=int, default=10)
    parser.add_argument("--total_steps", type=int, default=int(1e7))
    parser.add_argument("--rollout_len", type=int, default=125)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.1)
    parser.add_argument("--vf_coeff", type=float, default=0.5)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    args = parser.parse_args()
    return args


args = parse_args()
envs = gym.vector.SyncVectorEnv([
    create_env(args.env_name, args.seed+i) for i in range(args.actor_num)
])
eval_env = gym.make(args.env_name)
obs_dim = eval_env.observation_space.shape[0]
act_dim = eval_env.action_space.shape[0]


# PPO Storage
rollout_observations = torch.zeros((args.rollout_len, args.actor_num, obs_dim),
                                   dtype=torch.float32, device=device)
rollout_actions = torch.zeros((args.rollout_len, args.actor_num, act_dim),
                              dtype=torch.float32, device=device)
rollout_log_probs = torch.zeros((args.rollout_len, args.actor_num),
                                dtype=torch.float32).to(device)
rollout_rewards = torch.zeros((args.rollout_len, args.actor_num),
                              dtype=torch.float32, device=device)
rollout_dones = torch.zeros((args.rollout_len, args.actor_num),
                            dtype=torch.float32, device=device)
rollout_values = torch.zeros((args.rollout_len, args.actor_num),
                             dtype=torch.float32, device=device)


# PPO Agent
agent = Agent(obs_dim, act_dim).to(device)
optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)


# Steps setting
trajectory_len = args.actor_num * args.rollout_len
assert args.total_steps % trajectory_len == 0
loop_steps = args.total_steps // trajectory_len
batch_num = trajectory_len // args.batch_size
next_observations = torch.Tensor(envs.reset()).to(device)  # (10, 17)
next_dones = torch.Tensor(args.actor_num).to(device)


for step in trange(loop_steps):
    # Collecting new trajectories
    for i in range(args.rollout_len):
        rollout_observations[i] = next_observations
        rollout_dones[i] = next_dones
        with torch.no_grad():
            actions, log_probs = agent.sample_actions(next_observations)
            values = agent.get_values(next_observations)
        rollout_actions[i] = actions
        rollout_log_probs[i] = log_probs

        next_observations, rewards, dones, infos = envs.step(actions.cpu().numpy())
        next_observations = torch.Tensor(next_observations).to(device)
        next_dones = torch.Tensor(dones).to(device)

        rollout_rewards[i] = torch.Tensor(rewards).to(device).view(-1)

    # Compute GAE advantages & Cumulative returns
    with torch.no_grad():
        next_values = agent.get_values(next_observations).reshape(1, -1)  # (1, 10)
        advantages = torch.zeros_like(rollout_rewards).to(device)  # (125, 10)
        returns = torch.zeros_like(rollout_rewards).to(device)  # (125, 10) 
        last_gae = 0.
        for t in reversed(range(args.rollout_len)):
            if t == args.rollout_len - 1:
                next_nonterminal = 1. - next_dones
                next_returns = next_values
            else:
                next_nonterminal = 1. - rollout_dones[t+1]
                next_values = rollout_values[t+1]
                next_returns = rollout_rewards[t+1]
            delta = rollout_rewards[t] + args.gamma * next_nonterminal * next_values - rollout_values[t]
            advantages[t] = last_gae = \
                delta + args.gamma * args.lmbda * next_nonterminal * last_gae  # weighted pg_loss
            returns[t] = rollout_rewards[t] + args.gamma * next_nonterminal * next_returns
        # targets = advantages + rollout_values  # returns
        targets = returns

    # flatten
    traj_observations = rollout_observations.reshape(-1, obs_dim)
    traj_actions = rollout_actions.reshape(-1, act_dim)
    traj_log_probs = rollout_log_probs.reshape(-1)
    traj_targets = targets.reshape(-1)
    traj_advantages = advantages.reshape(-1)

    # train sampled trajectories for K epochs
    for _ in range(3):
        permutation = np.random.permutation(trajectory_len)
        for i in range(batch_num):
            batch_idx = permutation[i * args.batch_size:(i + 1) * args.batch_size]
            batch_observations=traj_observations[batch_idx]
            batch_actions=traj_actions[batch_idx]
            batch_log_probs=traj_log_probs[batch_idx]
            batch_targets=traj_targets[batch_idx]
            batch_advantages=traj_advantages[batch_idx]

            # training
            optimizer.zero_grad()
            new_log_probs, entropy = agent.get_logp(batch_observations, batch_actions)
            new_values = agent.get_values(batch_observations) 
            ratios = (new_log_probs - batch_log_probs).exp()

            # normalize GAE
            normalized_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-6)

            # pg loss
            pg_loss1 = normalized_advantages * ratios
            pg_loss2 = normalized_advantages * torch.clamp(
                ratios, 1.-args.clip_param, 1.+args.clip_param)
            pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

            # value loss
            value_loss = ((new_values - batch_targets)**2).mean() * args.vf_coeff

            # entropy loss
            entropy_loss = -entropy.mean() * args.entropy_coeff

            # total_loss
            total_loss = pg_loss + value_loss + entropy_loss
       
            total_loss.backward()
            optimizer.step()

            log_info = {
                "pg_loss": pg_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "total_loss": total_loss,
                "avg_target": batch_targets.mean(),
                "max_target": batch_targets.max(),
                "min_target": batch_targets.min(),
                "avg_value": new_values.mean(),
                "max_value": new_values.max(),
                "min_value": new_values.min(),
                "avg_ratio": ratios.mean(),
                "max_ratio": ratios.max(),
                "min_ratio": ratios.min(),
                "avg_logp": new_log_probs.mean(),
                "max_logp": new_log_probs.max(),
                "min_logp": new_log_probs.min(),
                "avg_delta_logp": (new_log_probs - batch_log_probs).mean(),
                "max_delta_logp": (new_log_probs - batch_log_probs).max(),
                "min_delta_logp": (new_log_probs - batch_log_probs).min(),
                "avg_old_logp": batch_log_probs.mean(),
                "max_old_logp": batch_log_probs.max(),
                "min_old_logp": batch_log_probs.min(),
                "a0": abs(batch_actions[:, 0]).mean(),
                "a1": abs(batch_actions[:, 1]).mean(),
                "a2": abs(batch_actions[:, 2]).mean(),
                "a3": abs(batch_actions[:, 3]).mean(),
                "a4": abs(batch_actions[:, 4]).mean(),
                "a5": abs(batch_actions[:, 5]).mean(),
            }

    if (step + 1) % 80 == 0:
        eval_reward, eval_step, eval_time = eval_policy(agent, eval_env)
        eval_fps = eval_step / eval_time
        print(
            f"\n#Step {step*trajectory_len}: eval_reward={eval_reward:.2f}, eval_time={eval_time:.2f}s, eval_fps={eval_fps:.2f}\n"
            # f"\ttotal_time={elapsed_time:.2f}min, step_fps={step_fps:.2f}\n"
            f"\tvalue_loss={log_info['value_loss']:.3f}, pg_loss={log_info['pg_loss']:.3f}, "
            f"entropy_loss={log_info['entropy_loss']:.3f}, total_loss={log_info['total_loss']:.3f}\n"
            f"\tavg_target={log_info['avg_target']:.3f}, max_target={log_info['max_target']:.3f}, min_target={log_info['min_target']:.3f}\n"
            f"\tavg_value={log_info['avg_value']:.3f}, max_value={log_info['max_value']:.3f}, min_value={log_info['min_value']:.3f}\n"
            f"\tavg_logp={log_info['avg_logp']:.3f}, max_logp={log_info['max_logp']:.3f}, min_logp={log_info['min_logp']:.3f}\n"
            f"\tavg_old_logp={log_info['avg_old_logp']:.3f}, max_old_logp={log_info['max_old_logp']:.3f}, min_old_logp={log_info['min_old_logp']:.3f}\n"
            f"\tavg_delta_logp={log_info['avg_delta_logp']:.3f}, max_delta_logp={log_info['max_delta_logp']:.3f}, min_delta_logp={log_info['min_delta_logp']:.3f}\n"
            f"\tavg_ratio={log_info['avg_ratio']:.3f}, max_ratio={log_info['max_ratio']:.3f}, min_ratio={log_info['min_ratio']:.3f}\n"
            f"\t(a0, a1, a2, a3, a4, a5) = ({log_info['a0']:.2f}, {log_info['a1']:.2f}, "
            f"{log_info['a2']:.2f}, {log_info['a3']:.2f}, {log_info['a4']:.2f}, "
            f"{log_info['a5']:.2f})\n"
        )
