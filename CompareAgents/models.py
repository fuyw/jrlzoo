from agents.iql import IQLAgent

from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
from flax import serialization
import distrax
import d4rl
import functools
import gym
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import time
from tqdm import trange

from dynamics_model import DynamicsModel
from utils import Batch


AGENT_DICT = {"iql": IQLAgent}


class CDAAgent:
    def __init__(self,
                 env_name: str,
                 algo: str,
                 obs_dim: int,
                 act_dim: int,
                 expectile: float = 0.9,
                 temperature: float = 10.0,
                 seed: int = 42,
                 horizon: int = 5,
                 var_thresh: float = -5.0,
                 initializer: str = "orthogonal",
                 rollout_batch_size: int = 10000):
        self.agent = AGENT_DICT[algo](obs_dim=obs_dim,
                                      act_dim=act_dim,
                                      expectile=expectile,
                                      temperature=temperature,
                                      initializer=initializer)
        self.dynamics_model = DynamicsModel(env_name=env_name)
        self.dynamics_model.load()
        self.env_name = env_name
        self.update_step = 0
        self.var_thresh = var_thresh
        self.rollout_batch_size = rollout_batch_size
        self.horizon = horizon
        self.rng = jax.random.PRNGKey(seed)

    def update_agent(self, replay_buffer, model_buffer, batch_size=128): 
        if self.update_step % 1000 == 0:
            observations = replay_buffer.sample(self.rollout_batch_size).observations
            for _ in range(self.horizon):
                self.rng, step_key = jax.random.split(self.rng, 2)
                actions = self.agent.sample_action(self.agent.actor_state.params, observations)
                normalized_observations = self.dynamics_model.normalize(observations)
                next_observations, rewards, dones, log_vars = self.dynamics_model.step(
                    step_key, normalized_observations, actions)
                if "v2" in self.env_name:
                    rewards = rewards / (replay_buffer.max_traj_reward - replay_buffer.min_traj_reward) * 1000 
                uncertainty_mask = log_vars < self.var_thresh
                model_buffer.add_batch(observations[uncertainty_mask],
                                       actions[uncertainty_mask],
                                       next_observations[uncertainty_mask],
                                       rewards[uncertainty_mask],
                                       dones[uncertainty_mask])
                nonterminal_mask = (~dones) & (log_vars < self.var_thresh)
                if nonterminal_mask.sum() == 0: break
                observations = next_observations[nonterminal_mask]

        # TODO: set minimum model buffer size
        if model_buffer.size > 200:
            real_batch = replay_buffer.sample(batch_size)
            model_batch = model_buffer.sample(batch_size)
            concat_batch = Batch(
                observations=np.concatenate([real_batch.observations, model_batch.observations], axis=0),
                actions = np.concatenate([real_batch.actions, model_batch.actions], axis=0),
                rewards = np.concatenate([real_batch.rewards, model_batch.rewards], axis=0),
                discounts = np.concatenate([real_batch.discounts, model_batch.discounts], axis=0),
                next_observations = np.concatenate([real_batch.next_observations, model_batch.next_observations], axis=0)
            )
        else:
            concat_batch = replay_buffer.sample(batch_size*2)
        self.update_step += 1
        log_info = self.agent.update(concat_batch)
        log_info["real_batch_obs"] = abs(real_batch.observations).sum(1).mean()
        log_info["model_batch_obs"] = abs(model_batch.observations).sum(1).mean()
        log_info["real_batch_act"] = abs(real_batch.actions).sum(1).mean()
        log_info["model_batch_act"] = abs(model_batch.actions).sum(1).mean()
        log_info["real_batch_reward"] = real_batch.rewards.mean()
        log_info["model_batch_reward"] = model_batch.rewards.mean()
        log_info["real_batch_discount"] = real_batch.discounts.sum()
        log_info["model_batch_discount"] = model_batch.discounts.sum()
        return log_info

    def sample_action(self, params, observation):
        sampled_action = self.agent.sample_action(params, observation)
        return sampled_action

    def save(self, fname: str, cnt: int):
        self.agent.save(fname, cnt)

    def load(self, fname: str, cnt: int):
        self.agent.load(fname, cnt)
