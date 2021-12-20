import bsuite
from bsuite import sweep
from bsuite.utils import gym_wrapper

import collections
import jax
import numpy as np

Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


class ReplayBuffer:
    def __init__(self, obs_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, 1), dtype=np.int32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.discounts = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, observation: np.ndarray, action: np.ndarray,
            next_observation: np.ndarray, reward: float, done: float):
        self.observations[self.ptr] = observation.flatten()
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation.flatten()
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


def evaluate_catch(agent):
    # 5 seeds for 5 different initial 
    seeds = [0, 1, 2, 3, 10]
    avg_rewards = 0
    start_positions = []
    for seed in seeds:
        sweep.SETTINGS["catch/0"]["seed"] = seed
        bsuite_env = bsuite.load_from_id("catch/0")
        env = gym_wrapper.GymFromDMEnv(bsuite_env)
        obs, done = env.reset(), False
        start_positions.append(obs[0].argmax())
        episode_reward = 0
        episode_timesteps = 0
        while not done:
            action = agent.select_action(agent.state.params, obs.flatten())
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            episode_reward += reward
            episode_timesteps += 1
        assert episode_timesteps == 9
        avg_rewards += episode_reward
    assert len(set(start_positions)) == 5
    avg_rewards /= 5
    return avg_rewards
