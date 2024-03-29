import collections
import functools
import logging
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import optax

ExpTuple = collections.namedtuple(
    "ExpTuple",
    ["observation", "action", "reward", "value", "log_prob", "done"])


Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "log_probs", "targets", "advantages"])



@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(rewards: np.ndarray, discounts: np.ndarray,
                   values: np.ndarray, gamma: float, lmbda: float):
    """Use GAE to compute advantages.

    As defined by Eq. (11-12) in PPO paper. Implementation uses key observations:
        A_{t} = delta_t + gamma * lmbda * A_{t+1},
              = delta_t + (gamma * lmbda) * delta_{t+1} + + (gamma * lmbda)^2 * delta_{t+2} + ...
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    """
    assert rewards.shape[0] + 1 == values.shape[0], ("One more value needed.")
    advantages = []
    gae = 0.
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal observations.
        value_diff = gamma * values[t + 1] * discounts[t] - values[t]
        delta = rewards[t] + value_diff

        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + gamma * lmbda * discounts[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, rollout_len, actor_num, gamma, lmbda):
        self.observations = np.zeros((rollout_len, actor_num, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_len, actor_num, act_dim), dtype=np.float32)
        self.rewards = np.zeros((rollout_len, actor_num), dtype=np.float32)
        self.values = np.zeros((rollout_len + 1, actor_num), dtype=np.float32)
        self.log_probs = np.zeros((rollout_len, actor_num), dtype=np.float32)
        self.discounts = np.zeros((rollout_len, actor_num), dtype=np.float32)

        self.ptr = 0
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.lmbda = lmbda
        self.actor_num = actor_num
        self.trajectory_len = actor_num * rollout_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def add(self, experience: List[ExpTuple]):
        for actor_id, actor_exp in enumerate(experience):
            self.observations[self.ptr, actor_id] = actor_exp.observation
            self.actions[self.ptr, actor_id] = actor_exp.action
            self.rewards[self.ptr, actor_id] = actor_exp.reward
            self.values[self.ptr, actor_id] = actor_exp.value
            self.log_probs[self.ptr, actor_id] = actor_exp.log_prob
            self.discounts[self.ptr, actor_id] = float(not actor_exp.done)
        self.ptr += 1

    def add_experiences(self, experiences: List[List[ExpTuple]]):
        assert len(experiences) == self.rollout_len + 1
        for experience in experiences[:-1]:
            self.add(experience)
        # experience[-1] for next_values
        for a in range(self.actor_num):
            self.values[-1, a] = experiences[-1][a].value
        self.ptr = 0

    def process_experience(self):
        # compute GAE advantage
        advantages = gae_advantages(self.rewards, self.discounts, self.values,
                                    self.gamma, self.lmbda)
        targets = advantages + self.values[:-1, :]

        # concatenate results
        trajectory_batch = Batch(
            observations=self.observations.reshape((self.trajectory_len, self.obs_dim)),
            actions=self.actions.reshape(self.trajectory_len, self.act_dim),
            log_probs=self.log_probs.reshape(self.trajectory_len,),
            targets=targets.reshape(self.trajectory_len,),
            advantages=advantages.reshape(self.trajectory_len,),
        )
        return trajectory_batch
