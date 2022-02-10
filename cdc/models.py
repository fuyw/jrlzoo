from typing import Any, Optional
import functools
from flax import linen as nn
from flax import serialization
from flax.core import FrozenDict
from flax.training import train_state
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from utils import Batch, ReplayBuffer

LOG_STD_MAX = 2.
LOG_STD_MIN = -5.

kernel_initializer = jax.nn.initializers.glorot_uniform()


def atanh(x: jnp.ndarray):
    one_plus_x = jnp.clip(1 + x, a_min=1e-6)
    one_minus_x = jnp.clip(1 - x, a_min=1e-6)
    return 0.5 * jnp.log(one_plus_x / one_minus_x)


class MLP(nn.Module):
    num_layers: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for i in range(self.num_layers):
            x = nn.relu(nn.Dense(256, kernel_init=kernel_initializer, name=f"fc{i+1}")(x))
        output = nn.Dense(self.out_dim, kernel_init=kernel_initializer, name="output")(x)
        return output


class Actor(nn.Module):
    act_dim: int
    hid_layers: int = 3
    action_limit: float = 1.0

    def setup(self):
        self.mlp = MLP(num_layers=self.hid_layers, out_dim=2*self.act_dim)

    def get_logprob(self, observation, action):
        x = self.mlp(observation)
        mu, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = distrax.Normal(mu, std)

        # Compute log_prob
        raw_action = atanh(action)
        log_prob = pi_distribution.log_prob(raw_action).sum(-1)
        log_prob -= (2 * jnp.log(2) - raw_action - jax.nn.softplus(-2 * raw_action)).sum(-1)

        return log_prob

    def __call__(self, observation: jnp.ndarray, seed: jnp.ndarray):
        x = self.mlp(observation)
        mu, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        mean_action = jnp.tanh(mu) * self.action_limit

        # Pre-squash distribution and sample
        pi_distribution = distrax.Normal(mu, std)
        pi_action = pi_distribution.sample(seed=seed) 
        squashed_actions = jnp.tanh(pi_action) * self.action_limit
        return mean_action, squashed_actions


class Critic(nn.Module):
    hid_dim: int = 256
    hid_layers: int = 3

    def setup(self):
        self.mlp = MLP(num_layers=self.hid_layers, out_dim=1)

    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observation, action], axis=-1)
        q = self.mlp(x)
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


class CDCAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 seed: int = 42,
                 nu: float = 1.0,
                 eta: float = 1.0,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lmbda: float = 1.0,
                 num_samples: int = 15,
                 lr: float = 7e-4,
                 cdc_sample: bool = False,
                 lr_actor: float = 3e-4):
        self.cdc_sample = cdc_sample
        self.update_step = 0
        self.tau = tau
        self.nu = nu
        self.eta = eta
        self.gamma = gamma
        self.num_samples = num_samples
        self.lmbda = lmbda
        self.lr = lr
        self.lr_actor = lr_actor

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(act_dim)
        actor_params = self.actor.init(actor_key, dummy_obs, actor_key)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.adam(self.lr_actor))

        # Initialize the Critic
        self.critic = MultiCritic()
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=Critic.apply,
            params=critic_params,
            tx=optax.adam(self.lr))
 
    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   critic_target_params: FrozenDict,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   key: jnp.ndarray):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        observations, actions, rewards, discounts, next_observations = batch
        def loss_fn(actor_params: FrozenDict,
                    critic_params: FrozenDict,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    discount: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    rng: jnp.ndarray):
            """compute loss for a single transition"""
            rng1, rng2, rng3 = jax.random.split(rng, 3)

            # Sample actions with Actor
            _, sampled_action = self.actor.apply({"params": actor_params}, observation, rng1)
            mle_prob = self.actor.apply(
                {"params": actor_params}, observation, action, method=self.actor.get_logprob)

            # We use frozen_params so that gradients can flow back to the actor without being
            # used to update the critic.
            concat_sampled_q = self.critic.apply(
                {"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = self.nu*concat_sampled_q.min(-1) + (1.-self.nu)*concat_sampled_q.max(-1)

            # Actor loss
            actor_loss = (-self.lmbda*mle_prob - sampled_q)

            # Critic loss
            concat_q = self.critic.apply({"params": critic_params}, observation, action)

            repeat_next_observations = jnp.repeat(
                jnp.expand_dims(next_observation, axis=0),
                repeats=self.num_samples, axis=0)
            _, sampled_next_actions = self.actor.apply({"params": frozen_actor_params},
                                                       repeat_next_observations, rng2)
            concat_next_q = self.critic.apply(
                {"params": critic_target_params}, repeat_next_observations, sampled_next_actions)
            weighted_next_q = self.nu * concat_next_q.min(-1) + (1. - self.nu) * concat_next_q.max(-1)
            next_q = weighted_next_q.max(-1)
            target_q = reward + self.gamma * discount * next_q
            critic_loss = jnp.square(concat_q - target_q).sum()

            # Overestimation penalty loss
            repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0),
                                             repeats=self.num_samples, axis=0)
            _, penalty_sampled_actions = self.actor.apply({"params": frozen_actor_params},
                                                          repeat_observations, rng3)
            penalty_concat_q = self.critic.apply(
                {"params": critic_params}, repeat_observations, penalty_sampled_actions).max(0)

            delta_concat_q = penalty_concat_q.reshape(-1, 1) - concat_q.reshape(1, -1)
            penalty_loss = jnp.square(jax.nn.relu(delta_concat_q)).mean()

            # logger info
            total_loss = critic_loss + actor_loss + penalty_loss * self.eta
            log_info = {
                "concat_q_avg": concat_q.mean(),
                "concat_q_min": concat_q.min(),
                "concat_q_max": concat_q.max(),
                "target_q": target_q,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "penalty_loss": penalty_loss,
                "mle_prob": mle_prob
            }
            return total_loss, log_info

        grad_fn = jax.vmap(
            jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True),
            in_axes=(None, None, 0, 0, 0, 0, 0, 0))
        rng = jnp.stack(jax.random.split(key, num=actions.shape[0]))

        (_, log_info), gradients = grad_fn(actor_state.params,
                                           critic_state.params,
                                           observations,
                                           actions,
                                           rewards,
                                           discounts,
                                           next_observations,
                                           rng)

        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)

        actor_grads, critic_grads = gradients

        # Update TrainState
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        return log_info, actor_state, critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def update_target_params(self, params: FrozenDict, target_params: FrozenDict):
        def _update(param, target_param):
            return self.tau * param + (1 - self.tau) * target_param

        updated_params = jax.tree_multimap(_update, params, target_params)
        return updated_params

    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray, eval_mode: bool = False) -> jnp.ndarray:
        rng, sample_rng = jax.random.split(rng)
        mean_action, sampled_action = self.actor.apply({"params": params}, observation, sample_rng)
        return rng, jnp.where(eval_mode, mean_action, sampled_action)
    
    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action2(self, actor_params: FrozenDict, critic_params: FrozenDict, rng: Any, observation: np.ndarray) -> jnp.ndarray:
        rng, sample_rng = jax.random.split(rng)
        repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0),
                                         repeats=self.num_samples, axis=0)
        _, sampled_actions = self.actor.apply({"params": actor_params}, repeat_observations, sample_rng)
        concat_qs = self.critic.apply({"params": critic_params}, repeat_observations, sampled_actions)
        weighted_q = self.nu * concat_qs.min(-1) + (1 - self.nu) * concat_qs.max(-1)
        max_idx = weighted_q.argmax()

        return rng,  sampled_actions[max_idx]

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        self.update_step += 1

        # sample from the buffer
        batch = replay_buffer.sample(batch_size)

        # train the network
        self.rng, key = jax.random.split(self.rng)
        log_info, self.actor_state, self.critic_state = self.train_step(
            batch, self.critic_target_params, self.actor_state, self.critic_state, key)

        # update target network
        params = self.critic_state.params
        target_params = self.critic_target_params
        self.critic_target_params = self.update_target_params(params, target_params)

        return log_info

    def save(self, filename):
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'wb') as f:
            f.write(serialization.to_bytes(self.critic_state.params))
        actor_file = filename + '_actor.ckpt'
        with open(actor_file, 'wb') as f:
            f.write(serialization.to_bytes(self.actor_state.params))

    def load(self, filename):
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'rb') as f:
            critic_params = serialization.from_bytes(
                self.critic_state.params, f.read())
        self.critic_state = train_state.TrainState.create(
            apply_fn=Critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=self.lr))

        actor_file = filename + '_actor.ckpt'
        with open(actor_file, 'rb') as f:
            actor_params = serialization.from_bytes(
                self.actor_state.params, f.read())
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=self.lr_actor)
        )
