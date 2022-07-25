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


def atanh(x: jnp.ndarray):
    one_plus_x = jnp.clip(1 + x, a_min=1e-6)
    one_minus_x = jnp.clip(1 - x, a_min=1e-6)
    return 0.5 * jnp.log(one_plus_x / one_minus_x)


class MLP(nn.Module):
    hidden_dims: Tuple[int] = (256, 256)
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
        self.out_layer = nn.Dense(1, kernel_init=init_fn(self.initializer, 1.0))

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
                                 kernel_init=init_fn(
                                     self.initializer,
                                     5/3))  # only affect orthogonal init
        self.std_layer = nn.Dense(self.act_dim,
                                  kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, rng: Any, observations: jnp.ndarray):
        x = self.net(observations)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        action_distributions = distrax.Normal(mu, std)
        sampled_actions, log_probs = action_distributions.sample_and_log_prob(seed=rng)
        return mu, sampled_actions, log_probs.sum(axis=-1)

    def get_logp(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        action_distributions = distrax.Normal(mu, std)
        log_probs = action_distributions.log_prob(actions).sum(axis=-1)
        entropy = action_distributions.entropy().sum(axis=-1)
        return log_probs, entropy


class ActorCritic(nn.Module):
    act_dim: int
    hidden_dims: Tuple[int] = (64, 64)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.actor = Actor(self.act_dim, hidden_dims=self.hidden_dims)
        self.critic = Critic(self.hidden_dims, self.initializer)

    def __call__(self, key: Any, observations: jnp.ndarray):
        mean_actions, sampled_actions, log_probs = self.actor(key, observations)
        values = self.critic(observations)
        return mean_actions, sampled_actions, log_probs, values

    def get_logp(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        log_probs, entropy = self.actor.get_logp(observations, actions)
        values = self.critic(observations)
        return log_probs, values, entropy


class PPOAgent:
    """PPOAgent adapted from Flax PPO example."""

    def __init__(self, config: ml_collections.ConfigDict, obs_dim: int, act_dim: int, lr: float):
        self.vf_coeff = config.vf_coeff
        self.entropy_coeff = config.entropy_coeff
        self.clip_param = config.clip_param

        # initialize learner
        self.rng = jax.random.PRNGKey(config.seed)
        dummy_obs = jnp.ones([1, obs_dim])
        self.learner = ActorCritic(act_dim=act_dim, hidden_dims=config.hidden_dims,
                                   initializer=config.initializer)
        learner_params = self.learner.init(self.rng, self.rng, dummy_obs)["params"]
        self.learner_state = train_state.TrainState.create(
            apply_fn=ActorCritic.apply,
            params=learner_params,
            tx=optax.adam(lr))

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self,
                       params: FrozenDict,
                       key: Any,
                       observations: jnp.ndarray,
                       eval_mode: bool = False):
        mean_actions, sampled_actions, log_probs, values = self.learner.apply(
            {"params": params}, key, observations)
        return jnp.where(eval_mode, mean_actions, sampled_actions), log_probs, values

    def sample_actions(self, observations: jnp.ndarray, eval_mode: bool = False):
        self.rng, key = jax.random.split(self.rng, 2)
        sampled_actions, log_probs, values = self._sample_action(
            self.learner_state.params, key, observations, eval_mode)
        sampled_actions, log_probs, values = jax.device_get((
            sampled_actions, log_probs, values))
        return sampled_actions.clip(-1., 1.), log_probs, values

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self, learner_state: train_state.TrainState, batch: Batch):

        def loss_fn(params, observations, actions, old_log_probs, targets, advantages):
            log_probs, values, entropy = self.learner.apply(
                {"params": params}, observations, actions, method=ActorCritic.get_logp)

            # clipped PPO loss
            ratios = jnp.exp(log_probs - old_log_probs)
            clipped_ratios = jnp.clip(ratios, 1. - self.clip_param, 1. + self.clip_param)
            actor_loss1 = ratios * advantages
            actor_loss2 = clipped_ratios * advantages
            ppo_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

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
                "avg_old_logp": old_log_probs.mean(),
                "max_old_logp": old_log_probs.max(),
                "min_old_logp": old_log_probs.min(),
            }
            return total_loss, log_info

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        normalized_advantages = (batch.advantages - batch.advantages.mean()
                                 ) / (batch.advantages.std() + 1e-8)
        (_, log_info), grads = grad_fn(learner_state.params,
                                       batch.observations, batch.actions,
                                       batch.log_probs, batch.targets,
                                       normalized_advantages)
        new_learner_state = learner_state.apply_gradients(grads=grads)
        return new_learner_state, log_info

    def update(self, batch: Batch):
        self.learner_state, log_info = self.train_step(self.learner_state,
                                                       batch)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname,
                                    self.learner_state,
                                    cnt,
                                    prefix="ppo_",
                                    keep=20,
                                    overwrite=True)

    def load(self, fname, step):
        self.learner_state = checkpoints.restore_checkpoint(
            ckpt_dir=fname,
            target=self.learner_state,
            step=step,
            prefix="ppo_")