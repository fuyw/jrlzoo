from collections import namedtuple
from typing import Callable

import jax
import numpy as np
from numpy.random import uniform


Batch = namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


HERBatch = namedtuple(
    "HERBatch",
    ["observations", "actions", "discounts", "next_observations", "goals" ,"rewards"])


class HERBuffer:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 goal_dim,
                 replay_k: int = 4,
                 traj_len: int = 100,
                 max_size: int = int(1e6),
                 reward_fn: Callable = None):
        # HER
        self.T = traj_len
        self.replay_k = replay_k
        self.future_p = 1 - (1. / (1 + replay_k))
        self.reward_fn = reward_fn

        # the idx to add next trajectory
        self.ptr = 0
        self.traj_num = 0
        self.max_traj_num = max_size // traj_len

        # trajectory buffer
        self.observations   = np.zeros([self.max_traj_num, self.T+1, obs_dim],
                                       dtype=np.float32)
        self.actions        = np.zeros([self.max_traj_num, self.T  , act_dim],
                                       dtype=np.float32)
        self.achieved_goals = np.zeros([self.max_traj_num, self.T+1, goal_dim],
                                       dtype=np.float32)
        self.goals          = np.zeros([self.max_traj_num, self.T  , goal_dim],
                                       dtype=np.float32)
        self.dones          = np.zeros([self.max_traj_num, self.T],
                                       dtype=np.float32)

    # save to trajectory buffer
    def add(self,
            traj_observations: np.ndarray,
            traj_achieved_goals: np.ndarray,
            traj_goals: np.ndarray,
            traj_actions: np.ndarray,
            traj_dones: np.ndarray):
        N = len(traj_actions)
        traj_idx = np.arange(self.ptr, self.ptr + N) % self.max_traj_num
        self.ptr = (self.ptr + N) % self.max_traj_num
        self.traj_num = min(self.max_traj_num, self.traj_num + N)
        self.observations[traj_idx] = traj_observations
        self.actions[traj_idx] = traj_actions
        self.achieved_goals[traj_idx] = traj_achieved_goals
        self.goals[traj_idx] = traj_goals
        self.dones[traj_idx] = traj_dones

    # sample from trajectory buffer
    def sample(self, batch_size):
        # sample a batch of data
        traj_idx = np.random.randint(self.traj_num, size=batch_size)
        step_idx = np.random.randint(self.T, size=batch_size)

        # HER idx
        her_idx = np.where(uniform(size=batch_size) < self.future_p)[0]
        future_offset = uniform(size=batch_size) * (self.T - step_idx)
        future_step_idx = (step_idx + 1 + future_offset.astype(int))[her_idx]

        # replace goal with future achieved goal
        goals = self.goals[traj_idx, step_idx].copy()
        goals[her_idx] = self.achieved_goals[traj_idx[her_idx],
                                             future_step_idx]

        # re-compute reward
        relabeled_rewards = self.reward_fn(self.achieved_goals[traj_idx, step_idx+1], goals, None)

        batch = HERBatch(
            observations=self.observations[traj_idx, step_idx],
            actions=self.actions[traj_idx, step_idx],
            discounts=1-self.dones[traj_idx, step_idx],
            next_observations=self.observations[traj_idx, step_idx+1],
            goals=goals,
            rewards=relabeled_rewards,
        )

        return batch
