from typing import Any, Optional
import functools
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
import os
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"


LOG_STD_MAX = 2.
LOG_STD_MIN = -5.

kernel_initializer = jax.nn.initializers.glorot_uniform()



class Actor(nn.Module):
    """return (mu, action_distribution)

    pi_action, logp_pi, lopprobs = self.actor(obs, gt_actions=actions, with_log_mle=True)
    """
    act_dim: int

    @nn.compact
    def __call__(self, rng: Any, observation: jnp.ndarray):
        x = nn.relu(nn.Dense(256, kernel_init=kernel_initializer, name="fc1")(observation))
        x = nn.relu(nn.Dense(256, kernel_init=kernel_initializer, name="fc2")(x))
        x = nn.relu(nn.Dense(256, kernel_init=kernel_initializer, name="fc3")(x))
        x = nn.Dense(2 * self.act_dim, kernel_init=kernel_initializer, name="output")(x)

        mu, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
 
        mean_action = jnp.tanh(mu)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1)
        )
        return mean_action, action_distribution
        # sampled_action, logp = action_distribution.sample_and_log_prob(seed=rng)
        # return mean_action, sampled_action, logp


class Critic(nn.Module):
    hid_dim: int = 256

    @nn.compact
    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observation, action], axis=-1)
        q = nn.relu(nn.Dense(self.hid_dim, kernel_init=kernel_initializer, name="fc1")(x))
        q = nn.relu(nn.Dense(self.hid_dim, kernel_init=kernel_initializer, name="fc2")(q))
        q = nn.relu(nn.Dense(self.hid_dim, kernel_init=kernel_initializer, name="fc3")(q))
        q = nn.Dense(1, kernel_init=kernel_initializer, name="output")(q)
        return q


class MultiCritic(nn.Module):
    hid_dim: int = 256

    def setup(self):
        self.critic1 = Critic(self.hid_dim)
        self.critic2 = Critic(self.hid_dim)
        self.critic3 = Critic(self.hid_dim)
        self.critic4 = Critic(self.hid_dim)

    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observation, action)
        q2 = self.critic2(observation, action)
        q3 = self.critic3(observation, action)
        q4 = self.critic4(observation, action)
        concat_q = jnp.concatenate([q1, q2, q3, q3], axis=-1)
        return concat_q


obs_dim = 11
act_dim = 3

rng = jax.random.PRNGKey(0)
rng, actor_key, critic_key = jax.random.split(rng, 3)
dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

actor = Actor(act_dim)
actor_params = actor.init(actor_key, actor_key, dummy_obs)["params"]
actor_state = train_state.TrainState.create(
    apply_fn=Actor.apply,
    params=actor_params,
    tx=optax.adam(1e-3))

critic = MultiCritic()
critic_params = critic.init(critic_key, dummy_obs, dummy_act)["params"]
critic_target_params = critic_params
critic_state = train_state.TrainState.create(
    apply_fn=Critic.apply,
    params=critic_params,
    tx=optax.adam(1e-3))

rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)
observation = np.random.normal(size=(obs_dim,))
sampled_action = np.random.normal(size=(act_dim,))
concat_q = critic.apply({"params": critic_params}, observation, sampled_action)  # (4,)



repeat_next_observations = jnp.repeat(jnp.expand_dims(observation, axis=0), repeats=15, axis=0)
_, next_pi_distribution = actor.apply({"params": actor_params}, rng2, repeat_next_observations)
sampled_next_actions = next_pi_distribution.sample(seed=jax.random.PRNGKey(4))  # (15, 3)

concat_next_q = critic.apply({"params": critic_params},
    repeat_next_observations, sampled_next_actions)  # (15, 4)

weighted_next_q = 0.5 * concat_next_q.min(-1) + 0.5 * concat_next_q.max(-1)  # (15,)
next_q = jnp.squeeze(weighted_next_q.max(-1))

target_q = 1.5 + 0.99 *  next_q

# Penalty
repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0), repeats=15, axis=0)  # (15, 11)
_, penalty_pi_distribution = actor.apply({"params": actor_params}, rng4, repeat_observations)
penalty_sampled_actions = penalty_pi_distribution.sample(seed=rng5)  # (15, 3)
penalty_concat_q = critic.apply({"params": critic_params}, repeat_observations, penalty_sampled_actions).max(0)  # (15, 4) ==> (4,)



