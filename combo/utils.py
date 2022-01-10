import collections
import jax
import numpy as np

Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


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


def get_training_data(replay_buffer, num_member=7, holdout_ratio=0.1):
    # load the offline data
    observations = replay_buffer.observations
    actions = replay_buffer.actions
    next_observations = replay_buffer.next_observations
    rewards = replay_buffer.rewards.reshape(-1, 1)  # reshape for correct shape
    delta_observations = next_observations - observations

    # prepare for model inputs & outputs
    inputs = np.concatenate([observations, actions], axis=-1)
    targets = np.concatenate([rewards, delta_observations], axis=-1)

    # validation dataset
    num_holdout = int(inputs.shape[0] * holdout_ratio)
    permutation = np.random.permutation(inputs.shape[0])

    # split the dataset
    inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
    targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
    holdout_inputs = np.tile(holdout_inputs[None], [num_member, 1, 1])
    holdout_targets = np.tile(holdout_targets[None], [num_member, 1, 1])

    return inputs, targets, holdout_inputs, holdout_targets
