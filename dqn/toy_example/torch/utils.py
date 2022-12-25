import collections
import torch

import numpy as np

import gym
from gym.envs.registration import registry, register


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(1e6)):
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
        batch = Batch(observations=torch.FloatTensor(self.observations[idx]).to(device),
                      actions=torch.LongTensor(self.actions[idx]).to(device),
                      rewards=torch.FloatTensor(self.rewards[idx]).to(device),
                      discounts=torch.FloatTensor(self.discounts[idx]).to(device),
                      next_observations=torch.FloatTensor(self.next_observations[idx]).to(device))
        return batch
    
    def save(self, fname):
        np.savez(fname,
                 observations=self.observations,
                 actions=self.actions,
                 rewards=self.rewards,
                 discounts=self.discounts,
                 next_observations=self.next_observations,
                 ptr=self.ptr,
                 size=self.size)


def register_custom_envs():
    if "PointmassEasy-v0" not in registry.env_specs:
        register(
            id="PointmassEasy-v0",
            entry_point="envs.pointmass.pointmass:Pointmass",
            kwargs={"difficulty": 0}
        )
    if "PointmassMedium-v0" not in registry.env_specs:
        register(
            id="PointmassMedium-v0",
            entry_point="envs.pointmass.pointmass:Pointmass",
            kwargs={"difficulty": 1}
        )
    if "PointmassHard-v0" not in registry.env_specs:
        register(
            id="PointmassHard-v0",
            entry_point="envs.pointmass.pointmass:Pointmass",
            kwargs={"difficulty": 2}
        )
    if "PointmassVeryHard-v0" not in registry.env_specs:
        register(
            id="PointmassVeryHard-v0",
            entry_point="envs.pointmass.pointmass:Pointmass",
            kwargs={"difficulty": 3}
        )
