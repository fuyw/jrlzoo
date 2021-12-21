import collections

import d4rl
import gym
import numpy as np
from tqdm import tqdm


Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])



class Dataset:
    def __int__(self,
                observations: np.ndarray,
                actions: np.ndarray,
                rewards: np.ndarray,
                discounts: np.ndarray,
                dones_float: np.ndarray,
                next_observations: np.ndarray,
                size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.discounts = discounts
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[idx],
                     actions=self.actions[idx],
                     rewards=self.rewards[idx],
                     discounts=self.discounts[idx],
                     next_observations=self.next_observations[idx])


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset["observations"][i + 1] -
                    dataset["next_observations"][i]) > 1e-6 or dataset["terminals"][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(observations=dataset["observations"].astype(np.float32),
                         actions=dataset["actions"].astype(np.float32))
