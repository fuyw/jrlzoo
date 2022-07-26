import functools
from typing import Any, Callable, Tuple

import distrax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import checkpoints, train_state

from utils import Batch

LOG_STD_MAX = 2.
LOG_STD_MIN = -10.


def init_fn(initializer: str, gain: float = jnp.sqrt(2)):
    if initializer == "orthogonal":
        return nn.initializers.orthogonal(gain)
    elif initializer == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    elif initializer == "glorot_normal":
        return nn.initializers.glorot_normal()
    return nn.initializers.lecun_normal()


class MLP(nn.Module):
    hidden_dims: Tuple[int] = (64, 64)
    init_fn: Callable = nn.initializers.glorot_uniform()
    activate_final: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.init_fn)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.tanh(x)
        return x


class Critic(nn.Module):
    hidden_dims: Tuple[int] = (64, 64)
    initializer: str = "glorot_uniform"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.out_layer = nn.Dense(1,
                                  kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        q = self.out_layer(x)
        return q.squeeze(-1)


class Actor(nn.Module):
    act_dim: int
    hidden_dims: Tuple[int] = (64, 64)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim,
                                 kernel_init=init_fn(self.initializer, 5/3))  # only affect orthogonal init
        self.log_std = self.param('log_std', nn.initializers.zeros, (self.act_dim,))

    def __call__(self, rng: Any, observations: jnp.ndarray):
        x = self.net(observations)
        mu = self.mu_layer(x)
        mu = nn.tanh(mu)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        action_distributions = distrax.MultivariateNormalDiag(mu, std)
        sampled_actions, log_probs = action_distributions.sample_and_log_prob(
            seed=rng)
        return mu, sampled_actions, log_probs

    def get_logp(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        mu = self.mu_layer(x)
        mu = nn.tanh(mu)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        action_distributions = distrax.MultivariateNormalDiag(mu, std)
        log_probs = action_distributions.log_prob(actions)
        entropy = action_distributions.entropy()
        return log_probs, entropy


class PPOAgent:
    """PPOAgent adapted from Flax PPO example."""

    def __init__(self, config: ml_collections.ConfigDict, obs_dim: int,
                 act_dim: int, lr: float):
        self.vf_coeff = config.vf_coeff
        self.entropy_coeff = config.entropy_coeff
        self.clip_param = config.clip_param

        # initialize learner
        self.rng = jax.random.PRNGKey(config.seed)
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)
        dummy_obs = jnp.ones([1, obs_dim])

        self.actor = Actor(act_dim, config.hidden_dims, config.initializer)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply, params=actor_params, tx=optax.adam(config.lr))

        self.critic = Critic(config.hidden_dims, config.initializer)
        critic_params = self.critic.init(critic_key, dummy_obs)["params"]
        self.critic_state = train_state.TrainState.create(
            apply_fn=Critic.apply, params=critic_params, tx=optax.adam(config.lr))

    @functools.partial(jax.jit, static_argnames=("self", "eval_mode"))
    def _sample_action(self,
                       params: FrozenDict,
                       key: Any,
                       observations: jnp.ndarray,
                       eval_mode: bool = False):
        mean_actions, sampled_actions, log_probs = self.actor.apply(
            {"params": params}, key, observations)
        return jnp.where(eval_mode, mean_actions, sampled_actions), log_probs

    def sample_actions(self,
                       observations: jnp.ndarray,
                       eval_mode: bool = False):
        self.rng, key = jax.random.split(self.rng, 2)
        sampled_actions, log_probs = self._sample_action(
            self.actor_state.params, key, observations, eval_mode)
        sampled_actions, log_probs = jax.device_get((sampled_actions, log_probs))
        return sampled_actions, log_probs

    @functools.partial(jax.jit, static_argnames=("self"))
    def _get_values(self, params, observations):
        values = self.critic.apply({"params": params}, observations)
        return values

    def get_values(self, observations: jnp.ndarray):
        values = self._get_values(self.critic_state.params, observations)
        return jax.device_get(values)

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   batch: Batch):

        def loss_fn(actor_params,
                    critic_params,
                    observations,
                    actions,
                    old_log_probs,
                    targets,
                    advantages):
            log_probs, entropy = self.actor.apply({"params": actor_params},
                                                  observations,
                                                  actions,
                                                  method=Actor.get_logp) 
            values = self.critic.apply({"params": critic_params}, observations)

            # clipped PPO loss
            ratios = jnp.exp(log_probs - old_log_probs)
            clipped_ratios = jnp.clip(ratios, 1. - self.clip_param,
                                      1. + self.clip_param)
            pg_loss1 = ratios * advantages
            pg_loss2 = clipped_ratios * advantages
            ppo_loss = -jnp.minimum(pg_loss1, pg_loss2).mean()

            # value loss
            value_loss = jnp.square(targets - values).mean() * self.vf_coeff

            # entropy loss
            entropy_loss = -entropy.mean() * self.entropy_coeff

            # total loss
            total_loss = ppo_loss + value_loss + entropy_loss

            log_info = {
                "ppo_loss": ppo_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "total_loss": total_loss,
                "avg_target": targets.mean(),
                "max_target": targets.max(),
                "min_target": targets.min(),
                "avg_value": values.mean(),
                "max_value": values.max(),
                "min_value": values.min(),
                "avg_ratio": ratios.mean(),
                "max_ratio": ratios.max(),
                "min_ratio": ratios.min(),
                "avg_logp": log_probs.mean(),
                "max_logp": log_probs.max(),
                "min_logp": log_probs.min(),
                "avg_delta_logp": (log_probs - old_log_probs).mean(),
                "max_delta_logp": (log_probs - old_log_probs).max(),
                "min_delta_logp": (log_probs - old_log_probs).min(),
                "avg_old_logp": old_log_probs.mean(),
                "max_old_logp": old_log_probs.max(),
                "min_old_logp": old_log_probs.min(),
                "a0": actions[:, 0].mean(),
                "a1": actions[:, 1].mean(),
                "a2": actions[:, 2].mean(),
                "a3": actions[:, 3].mean(),
                "a4": actions[:, 4].mean(),
                "a5": actions[:, 5].mean(),
            }
            return total_loss, log_info

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
        normalized_advantages = (batch.advantages - batch.advantages.mean()
                                 ) / (batch.advantages.std() + 1e-8)
        (_, log_info), grads = grad_fn(actor_state.params,
                                       critic_state.params,
                                       batch.observations,
                                       batch.actions,
                                       batch.log_probs,
                                       batch.targets,
                                       normalized_advantages)
        actor_grads, critic_grads = grads
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)
        return new_actor_state, new_critic_state, log_info

    def update(self, batch: Batch):
        self.actor_state, self.critic_state, log_info = self.train_step(
            self.actor_state, self.critic_state, batch)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname,
                                    self.actor_state,
                                    cnt,
                                    prefix="ppo_actor_",
                                    keep=20,
                                    overwrite=True)
        checkpoints.save_checkpoint(fname,
                                    self.critic_state,
                                    cnt,
                                    prefix="ppo_critic_",
                                    keep=20,
                                    overwrite=True)

    def load(self, fname, step):
        self.actor_state = checkpoints.restore_checkpoint(
            ckpt_dir=fname,
            target=self.actor_state,
            step=step,
            prefix="ppo_actor_")
        self.critic_state = checkpoints.restore_checkpoint(
            ckpt_dir=fname,
            target=self.critic_state,
            step=step,
            prefix="ppo_critic_")
