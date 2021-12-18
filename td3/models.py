from typing import Any, Tuple

import functools

from flax import linen as nn
from flax import serialization
from flax.core import frozen_dict
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils import Batch, ReplayBuffer


class Actor(nn.Module):
    act_dim: int
    max_action: float

    def setup(self):
        self.l1 = nn.Dense(256, name="fc1")
        self.l2 = nn.Dense(256, name="fc2")
        self.l3 = nn.Dense(self.act_dim, name="fc3")

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(self.l1(observations))
        x = nn.relu(self.l2(x))
        actions = self.max_action * nn.tanh(self.l3(x))
        return actions


class Critic(nn.Module):
    def setup(self):
        self.l1 = nn.Dense(256, name="fc1")
        self.l2 = nn.Dense(256, name="fc2")
        self.l3 = nn.Dense(1, name="fc3")

        self.l4 = nn.Dense(256, name="fc4")
        self.l5 = nn.Dense(256, name="fc5")
        self.l6 = nn.Dense(1, name="fc6")

    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observations, actions], axis=-1)

        q1 = nn.relu(self.l1(x))
        q1 = nn.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = nn.relu(self.l4(x))
        q2 = nn.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, observations: jnp.ndarray,
           actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        q1 = nn.relu(self.l1(x))
        q1 = nn.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 noise_clip: float = 0.5,
                 policy_noise: float = 0.2,
                 policy_freq: int = 2,
                 learning_rate: float = 3e-4,
                 seed: int = 42):

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        rng = jax.random.PRNGKey(seed)
        self.actor_rng, self.critic_rng = jax.random.split(rng, 2)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Initialize the actor
        self.actor = Actor(act_dim, max_action)
        actor_params = self.actor.init(self.actor_rng, dummy_obs)["params"]
        self.actor_target_params = actor_params
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=learning_rate))

        # Initialize the critic
        self.critic = Critic()
        critic_params = self.critic.init(self.critic_rng, dummy_obs,
                                         dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=Critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=learning_rate))

        self.update_step = 0

    @functools.partial(jax.jit, static_argnames=("self"))
    def actor_train_step(self, actor_state: train_state.TrainState,
                         critic_state: train_state.TrainState,
                         observations: jnp.ndarray):
        def loss_fn(actor_params, critic_params):
            actions = self.actor.apply({"params": actor_params}, observations)
            q_val = self.critic.apply({"params": critic_params},
                                      observations,
                                      actions,
                                      method=Critic.Q1)
            actor_loss = -jnp.mean(q_val)
            return actor_loss

        actor_loss, actor_grads = jax.value_and_grad(loss_fn, argnums=0)(
            actor_state.params, critic_state.params)

        # Update Actor TrainState
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        actor_info = {"actor_loss": actor_loss}
        return actor_info, actor_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def critic_train_step(self, batch: Batch, critic_key: Any,
                          critic_state: train_state.TrainState,
                          actor_target_params: frozen_dict.FrozenDict,
                          critic_target_params: frozen_dict.FrozenDict):
        # Add noise to actions
        noise = jax.random.normal(critic_key,
                                  batch.actions.shape) * self.policy_noise
        noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)
        next_actions = self.actor.apply({"params": actor_target_params},
                                        batch.next_observations)
        next_actions = jnp.clip(next_actions + noise, -self.max_action,
                                self.max_action)

        # Compute the target Q value
        next_q1, next_q2 = self.critic.apply({"params": critic_target_params},
                                             batch.next_observations,
                                             next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        target_q = batch.rewards + self.gamma * batch.discounts * next_q

        def loss_fn(critic_params: frozen_dict.FrozenDict, batch: Batch):
            q1, q2 = self.critic.apply({"params": critic_params},
                                       batch.observations, batch.actions)
            critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean()
            }

        (critic_loss, critic_info), critic_grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True)(critic_state.params, batch)

        #  update Critic TrainState
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnums=0)
    def update_target_params(self, params: frozen_dict.FrozenDict,
                             target_params: frozen_dict.FrozenDict):
        def _update(param, target_param):
            return self.tau * param + (1 - self.tau) * target_param

        updated_params = jax.tree_multimap(_update, params, target_params)
        return updated_params

    @functools.partial(jax.jit, static_argnums=0)
    def select_action(self, params: frozen_dict.FrozenDict,
                      observations: np.ndarray) -> jnp.ndarray:
        observations = jax.device_put(observations[None])
        actions = self.actor.apply({"params": params}, observations)
        actions = actions.flatten()
        return actions

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        self.update_step += 1

        # Sample replay buffer
        batch = replay_buffer.sample(batch_size)

        # Critic update
        self.critic_rng, critic_key = jax.random.split(self.critic_rng)
        critic_info, self.critic_state = self.critic_train_step(
            batch, critic_key, self.critic_state, self.actor_target_params,
            self.critic_target_params)

        # Delayed policy update
        if self.update_step % self.policy_freq == 0:
            actor_info, self.actor_state = self.actor_train_step(
                self.actor_state, self.critic_state, batch.observations)

            # update target network
            params = (self.actor_state.params, self.critic_state.params)
            target_params = (self.actor_target_params,
                             self.critic_target_params)
            updated_params = self.update_target_params(params, target_params)
            self.actor_target_params, self.critic_target_params = updated_params

        return critic_info

    def save(self, filename):
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'wb') as f:
            f.write(serialization.to_bytes(self.critic_state.params))
        actor_file = filename + '_actor.ckpt'
        with open(actor_file, 'wb') as f:
            f.write(serialization.to_bytes(self.actor_state.params))

    def load(self, filename):
        # TODO: model loading is untested
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'rb') as f:
            self.critic_params = serialization.from_bytes(
                self.critic_params, f.read())
        self.critic_target_params = self.critic_params
        actor_file = filename + '_actor.ckpt'
        with open(actor_file, 'rb') as f:
            self.actor_params = serialization.from_bytes(
                self.actor_params, f.read())
        self.actor_target_params = self.actor_params
