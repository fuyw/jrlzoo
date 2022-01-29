import collections
import jax
import numpy as np

Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


QBatch = collections.namedtuple(
    "QBatch",
    ["observations", "actions", "rewards", "discounts", "next_observations", "qpos", "qvel"])


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)

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
        batch = Batch(observations=jax.device_put(self.observations[idx]),
                      actions=jax.device_put(self.actions[idx]),
                      rewards=jax.device_put(self.rewards[idx]),
                      discounts=jax.device_put(self.discounts[idx]),
                      next_observations=jax.device_put(
                          self.next_observations[idx]))
        return batch

    def convert_D4RL(self, dataset):
        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.next_observations = dataset["next_observations"]
        self.rewards = dataset["rewards"]
        self.discounts = 1. - dataset["terminals"]
        self.size = self.observations.shape[0]

    def normalize_states(self, eps: float = 1e-3):
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean)/std
        self.next_observations = (self.next_observations - mean)/std
        return mean, std


class InfoBuffer:
    def __init__(self, obs_dim: int, act_dim: int, qpos_dim: int = None,
                 qvel_dim: int = None, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)
        self.qpos = np.zeros((max_size, qpos_dim))
        self.qvel = np.zeros((max_size, qvel_dim))

    def add(self, observation: np.ndarray, action: np.ndarray,
            next_observation: np.ndarray, reward: float, done: float,
            qpos: np.ndarray, qvel: np.ndarray):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.qpos[self.ptr] = qpos
        self.qvel[self.ptr] = qvel

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = QBatch(observations=jax.device_put(self.observations[idx]),
                       actions=jax.device_put(self.actions[idx]),
                       rewards=jax.device_put(self.rewards[idx]),
                       discounts=jax.device_put(self.discounts[idx]),
                       next_observations=jax.device_put(self.next_observations[idx]),
                       qpos=self.qpos[idx],
                       qvel=self.qvel[idx])
        return batch

    def save(self, fname):
        np.savez(fname,
                 observations=self.observations[:self.size],
                 actions=self.actions[:self.size],
                 next_observations=self.next_observations[:self.size],
                 rewards=self.rewards[:self.size],
                 discounts=self.discounts[:self.size],
                 qpos=self.qpos[:self.size],
                 qvel=self.qvel[:self.size])

    def load(self, fname='saved_buffers/Hopper-v2/s0.npz'):
        dataset = np.load(fname)
        self.observations = dataset['observations']
        self.actions = dataset['actions']
        self.next_observations = dataset['next_observations']
        self.rewards = dataset['rewards'].reshape(-1)
        self.discounts = dataset['discounts'].reshape(-1)
        self.qpos = dataset['qpos']
        self.qvel = dataset['qvel']
        self.size = len(self.actions)
