from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import copy
import functools

import numpy as np

import distrax
import optax
import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state
from flax.core import FrozenDict

from utils import init_fn, random_crop, batched_random_crop, target_update


###################
# Utils Functions #
###################
class MLP(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    init_fn: Callable = nn.initializers.glorot_uniform()
    activate_final: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.init_fn)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


class Encoder(nn.Module):
    features: Sequence[int] = (32, 64, 128, 256)
    kernels: Sequence[int] = (3, 3, 3, 3)
    strides: Sequence[int] = (2, 2, 2, 2)
    init_fn: Callable = nn.initializers.glorot_uniform()
    padding: str = "VALID"

    @nn.compact
    def __call__(self, observations):
        # (64, 64, 3, 3)
        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for features, kernel, stride in zip(self.features,
                                            self.kernels,
                                            self.strides):
            x = nn.Conv(features,
                        kernel_size=(kernel, kernel),
                        strides=(stride, stride),
                        padding=self.padding,
                        kernel_init=self.init_fn)(x)
            x = nn.relu(x)
        return x.reshape((*x.shape[:-3], -1))


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


#####################
# Soft Actor Critic #
#####################
class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"
    output_dim: int = 1

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.out_layer = nn.Dense(self.output_dim, kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze()


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"
    output_dim: int = 1
    num_qs: int = 2

    @nn.compact
    def __call__(self, observations, actions, training: bool = False):
        VmapCritic = nn.vmap(Critic,
                             variable_axes={"params": 0},
                             split_rngs={"params": True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        self.initializer,
                        self.output_dim)(observations, actions)
        return qs


class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"
    min_scale: float = 1e-3

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim, kernel_init=init_fn(self.initializer, 5/3))
        self.std_layer = nn.Dense(self.act_dim, kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, rng: Any, observation: jnp.ndarray):
        x = self.net(observation)
        mu = self.mu_layer(x)
        mean_action = nn.tanh(mu)

        std = self.std_layer(x)
        std = jax.nn.softplus(std) + self.min_scale

        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))

        sampled_action, logp = action_distribution.sample_and_log_prob(seed=rng)

        return mean_action * self.max_action, sampled_action * self.max_action, logp


class DrQCritic(nn.Module):
    encoder: nn.Module
    critic: nn.Module
    emb_dim: int = 50
    initializer: str = "orthogonal"

    @nn.compact
    def __call__(self, observations, actions):
        x = self.encoder(observations)
        x = nn.Dense(self.emb_dim, kernel_init=init_fn(self.initializer))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        return self.critic(x, actions)


class DrQActor(nn.Module):
    encoder: nn.Module
    actor: nn.Module
    emb_dim: int = 50
    initializer: str = "orthogonal"

    @nn.compact
    def __call__(self, rng, observations):
        x = self.encoder(observations)
        x = jax.lax.stop_gradient(x)
        x = nn.Dense(self.emb_dim, kernel_init=init_fn(self.initializer))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        return self.actor(rng, x)


#############
# DrQ Agent #
#############
class DrQAgent:
    def __init__(self,
                 obs_shape,
                 act_dim,
                 max_action: float = 1.0,
                 emb_dim: int = 50,
                 seed: int = 42,
                 lr: float = 3e-4,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 64, 128, 256),
                 cnn_kernels: Sequence[int] = (3, 3, 3, 3),
                 cnn_strides: Sequence[int] = (2, 2, 2, 2),
                 cnn_padding: str = "VALID",
                 initializer: str = "orthogonal",
                 target_entropy: Optional[float] = None):

        self.tau = tau
        self.gamma = gamma
        self.max_action = max_action
        if target_entropy is None:
            self.target_entropy = -act_dim / 2
        else:
            self.target_entropy = target_entropy

        dummy_obs = jnp.ones((1, *obs_shape), dtype=jnp.float32)
        dummy_act = jnp.ones((1, act_dim), dtype=jnp.float32)

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key, alpha_key = jax.random.split(self.rng, 4)

        # Encoder
        encoder = Encoder(features=cnn_features,
                          kernels=cnn_kernels,
                          strides=cnn_strides,
                          padding=cnn_padding,
                          init_fn=init_fn(initializer))

        # Critic
        sac_critic = DoubleCritic(hidden_dims=hidden_dims,
                                  initializer=initializer)
        self.critic = DrQCritic(encoder=encoder,
                                critic=sac_critic,
                                emb_dim=emb_dim,
                                initializer=initializer)
        critic_params = self.critic.init(critic_key,
                                         dummy_obs,
                                         dummy_act)["params"]
        self.target_critic_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=lr))

        # Actor
        sac_actor = Actor(act_dim=act_dim,
                          hidden_dims=hidden_dims,
                          initializer=initializer)
        self.actor = DrQActor(encoder=encoder,
                              actor=sac_actor,
                              emb_dim=emb_dim,
                              initializer=initializer)
        actor_params = self.actor.init(actor_key,
                                       actor_key,
                                       dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=lr))

        # Entropy
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(self.rng)["params"],
            tx=optax.adam(1e-3))

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self,
                       params: FrozenDict,
                       rng: Any,
                       observation: np.ndarray) -> jnp.ndarray:
        mean_action, sampled_action, _ = self.actor.apply(
            {"params": params}, rng, observation)
        return mean_action, sampled_action

    def sample_action(self, observation, eval_mode: bool = False):
        self.rng, sample_rng = jax.random.split(self.rng)
        mean_action, sampled_action = self._sample_action(
            self.actor_state.params,
            sample_rng,
            observation["pixels"])
        action = mean_action if eval_mode else sampled_action
        action = np.asarray(action)
        return action.clip(-self.max_action, self.max_action)

    def actor_alpha_train_step(self,
                               observations: jnp.ndarray,
                               key: Any,
                               alpha_state: train_state.TrainState,
                               actor_state: train_state.TrainState,
                               critic_state: train_state.TrainState):

        def actor_loss_fn(alpha_params: FrozenDict, actor_params: FrozenDict):
            _, actions, log_probs = self.actor.apply({"params": actor_params},
                                                     key,
                                                     observations)

            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha = jnp.exp(log_alpha)
            alpha_loss = -alpha * jax.lax.stop_gradient(log_probs + self.target_entropy).mean()
            alpha = jax.lax.stop_gradient(alpha)

            qs = self.critic.apply({"params": critic_state.params}, observations, actions)
            q = qs.mean(axis=0)
            actor_loss = (alpha * log_probs - q).mean()
            total_loss = alpha_loss + actor_loss
            return total_loss, {
                "alpha": alpha,
                "logp": log_probs.mean(),
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "entropy": -log_probs.mean(),
            }

        grads, info = jax.grad(actor_loss_fn, argnums=(0, 1), has_aux=True)(alpha_state.params,
                                                                            actor_state.params)
        alpha_grads, actor_grads = grads
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)

        return new_alpha_state, new_actor_state, info

    def critic_train_step(self,
                          observations: jnp.ndarray,
                          actions: jnp.ndarray,
                          rewards: jnp.ndarray,
                          next_observations: jnp.ndarray,
                          discounts: jnp.ndarray,
                          key: Any,
                          alpha_state: train_state.TrainState,
                          actor_state: train_state.TrainState,
                          critic_state: train_state.TrainState,
                          target_critic_params: FrozenDict):
        _, next_actions, next_log_probs = self.actor.apply(
            {"params": actor_state.params}, key, next_observations)
        next_qs = self.critic.apply({"params": target_critic_params},
                                    next_observations, next_actions)

        log_alpha = self.log_alpha.apply({"params": alpha_state.params})
        alpha = jnp.exp(log_alpha)

        next_q = next_qs.min(axis=0) - alpha * next_log_probs
        target_q = rewards + self.gamma * discounts * next_q

        def critic_loss_fn(critic_params):
            qs = self.critic.apply({"params": critic_params}, observations, actions)
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
                "target_q": target_q.mean(),
            }
        grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=grads)
        new_critic_target_params = target_update(new_critic_state.params,
                                                 target_critic_params,
                                                 self.tau)
        return new_critic_state, new_critic_target_params, info

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch,
                   rng,
                   alpha_state,
                   actor_state,
                   critic_state,
                   target_critic_params):

        # update actor's encoder parameter
        sync_params = actor_state.params.copy(
            add_or_replace={"encoder": critic_state.params["encoder"]})
        actor_state = actor_state.replace(params=sync_params)

        # data augmentation
        aug_key1, aug_key2, actor_key, critic_key = jax.random.split(rng, 4)
        aug_observations = batched_random_crop(aug_key1, batch.observations[..., :-1])
        aug_next_observations = batched_random_crop(aug_key2, batch.observations[..., 1:])

        # update model
        (new_critic_state,
         new_target_critic_params,
         critic_info) = self.critic_train_step(aug_observations,
                                               batch.actions,
                                               batch.rewards,
                                               aug_next_observations,
                                               batch.discounts,
                                               critic_key,
                                               alpha_state,
                                               actor_state,
                                               critic_state,
                                               target_critic_params)

        (new_alpha_state,
         new_actor_state,
         actor_info) = self.actor_alpha_train_step(aug_observations,
                                                   actor_key,
                                                   alpha_state,
                                                   actor_state,
                                                   critic_state)

        return new_alpha_state, new_actor_state, new_critic_state, \
            new_target_critic_params, {**critic_info, **actor_info}

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        self.rng, key = jax.random.split(self.rng)
        (self.alpha_state,
         self.actor_state,
         self.critic_state,
         self.target_critic_params,
         info) = self.train_step(batch,
                                 key,
                                 self.alpha_state,
                                 self.actor_state,
                                 self.critic_state,
                                 self.target_critic_params)
        return info
