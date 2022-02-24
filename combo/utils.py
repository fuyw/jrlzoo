import collections
import jax
import numpy as np


Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])

QBatch = collections.namedtuple(
    "QBatch",
    ["observations", "actions", "rewards", "discounts", "next_observations", "qpos", "qvel"])


def get_training_data(replay_buffer, ensemble_num=7, holdout_num=50000, eps=1e-3):
    """
    inputs.shape  = (N, obs_dim+act_dim)
    targets.shape = (N, obs_dim+1) 
    holdout_inputs.shape  = (E, N, obs_dim+act_dim)
    holdout_targets.shape = (E, N, obs_dim+1)
    """
    # load the offline data
    observations = replay_buffer.observations
    actions = replay_buffer.actions
    next_observations = replay_buffer.next_observations
    rewards = replay_buffer.rewards.reshape(-1, 1)  # reshape for correct shape

    # validation dataset
    permutation = np.random.permutation(len(observations))
    train_idx, target_idx = permutation[:-holdout_num], permutation[-holdout_num:]

    # split validation set
    train_observations = observations[train_idx]
    train_actions = actions[train_idx]
    train_inputs = np.concatenate([train_observations, train_actions], axis=-1)

    # compute the normalize stats
    obs_mean = train_observations.mean(0, keepdims=True)
    obs_std = train_observations.std(0, keepdims=True) + eps

    # normlaize the data
    observations = (observations - obs_mean) / obs_std
    next_observations = (next_observations - obs_mean) / obs_std
    delta_observations = next_observations - observations

    # prepare for model inputs & outputs
    inputs = np.concatenate([observations, actions], axis=-1)
    targets = np.concatenate([rewards, delta_observations], axis=-1)

    # split the dataset
    inputs, holdout_inputs = inputs[train_idx], inputs[target_idx]
    targets, holdout_targets = targets[train_idx], targets[target_idx]
    holdout_inputs = np.tile(holdout_inputs[None], [ensemble_num, 1, 1])
    holdout_targets = np.tile(holdout_targets[None], [ensemble_num, 1, 1])

    return inputs, targets, holdout_inputs, holdout_targets, obs_mean, obs_std


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
    
    def add_batch(self, observations, actions, next_observations, rewards, dones):
        add_num = len(actions)
        add_idx = np.arange(self.ptr, self.ptr + add_num) % self.max_size
        self.observations[add_idx] = observations
        self.actions[add_idx] = actions
        self.next_observations[add_idx] = next_observations
        self.rewards[add_idx] = rewards
        self.discounts[add_idx] = 1 - dones
        self.ptr = (self.ptr + add_num) % self.max_size
        self.size = min(self.size + add_num, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=self.observations[idx],
                      actions=self.actions[idx],
                      rewards=self.rewards[idx],
                      discounts=self.discounts[idx],
                      next_observations=self.next_observations[idx])
        return batch

    def convert_D4RL(self, dataset):
        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.next_observations = dataset["next_observations"]
        self.rewards = dataset["rewards"]
        self.discounts = 1. - dataset["terminals"]
        self.size = self.observations.shape[0]
    
    def convert_D4RL2(self, dataset):
        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.next_observations = dataset["next_observations"]
        self.rewards = dataset["rewards"]
        self.discounts = 1. - dataset["terminals"]
        self.size = self.observations.shape[0]
        self.qpos = dataset["qpos"]
        self.qvel = dataset["qvel"]

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
