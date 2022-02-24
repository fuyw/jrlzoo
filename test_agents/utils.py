import collections
import jax
import numpy as np
from models.td3bc import TD3BCAgent

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
        self.rewards = dataset["rewards"].reshape(-1)
        self.discounts = 1. - dataset["terminals"].reshape(-1)
        self.size = self.observations.shape[0]

    def normalize_states(self, eps: float = 1e-3):
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean)/std
        self.next_observations = (self.next_observations - mean)/std
        return mean, std


def load_data(args):
    data = np.load(f"saved_buffers/{args.env_name.split('-')[0]}/5agents.npz")
    observations = data["observations"]
    actions = data["actions"]
    next_observations = data["next_observations"]
    rewards = data["rewards"]
    discounts = data["discounts"]
    return observations, actions, rewards, next_observations
 

def get_embeddings(args, agent, observations, actions):
    if isinstance(agent, TD3BCAgent):
        observations = (observations - mu)/std

    L = len(observations)
    batch_size = 10000
    batch_num = int(np.ceil(L / batch_size))
    encode = jax.jit(agent.encode)
    embeddings = []
    for i in trange(batch_num):
        batch_observations = observations[i*batch_size:(i+1)*batch_size]
        batch_actions = actions[i*batch_size:(i+1)*batch_size]
        batch_embedding = encode(batch_observations, batch_actions)
        embeddings.append(batch_embedding)

    embeddings = np.concatenate(embeddings, axis=0)
    assert len(embeddings) == L
    return embeddings
