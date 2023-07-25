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


###################
# Utils Functions #
###################
def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((2,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


def target_update(params, target_params, tau: float = 0.005):
    def _update(param: FrozenDict, target_param: FrozenDict):
        return tau * param + (1 - tau) * target_param
    updated_params = jax.tree_util.tree_map(_update, params, target_params)
    return updated_params


def _unpack(batch):
    obs_pixels = batch["observations"]["pixels"][..., :-1]
    next_obs_pixels = batch["observations"]["pixels"][..., 1:]
    obs = batch["observations"].copy(add_or_replace={"pixels": obs_pixels})
    next_obs = batch["next_observations"].copy(add_or_replace={"pixels": next_obs_pixels})
    batch = batch.copy(add_or_replace={"observations": obs, "next_observations": next_obs})
    return batch


def _share_encoder(source, target):
    new_params = target.params.copy(
        add_or_replace={"encoder": source.params["encoder"]}
    )
    return target.replace(params=new_params)


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
                        padding=self.padding)(x)
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
        self.net = MLP(self.hidden_dims, activate_final=True)
        self.out_layer = nn.Dense(self.output_dim)

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
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    min_scale: float = 1e-3

    def setup(self):
        self.net = MLP(self.hidden_dims, activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim)
        self.std_layer = nn.Dense(self.act_dim)

    def __call__(self, observation: jnp.ndarray, training: bool = False):
        x = self.net(observation)
        mu = self.mu_layer(x)
        mean_action = nn.tanh(mu)

        std = self.std_layer(x)
        std = jax.nn.softplus(std) + self.min_scale

        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))

        return mean_action * self.max_action, action_distribution


class DrQCritic(nn.Module):
    encoder: nn.Module
    critic: nn.Module
    emb_dim: int = 50

    @nn.compact
    def __call__(self, observations, actions):
        x = self.encoder(observations)
        x = nn.Dense(self.emb_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        return self.critic(x, actions)


class DrQActor(nn.Module):
    encoder: nn.Module
    actor: nn.Module
    emb_dim: int = 50

    @nn.compact
    def __call__(self, observations):
        x = self.encoder(observations)
        x = jax.lax.stop_gradient(x)
        x = nn.Dense(self.emb_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        return self.actor(x)


#############
# DrQ Agent #
#############
class DrQLearner:
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
                          padding=cnn_padding)

        # Critic
        sac_critic = DoubleCritic()
        self.critic = DrQCritic(encoder=encoder,
                                critic=sac_critic,
                                emb_dim=emb_dim)
        critic_params = self.critic.init(critic_key,
                                         dummy_obs,
                                         dummy_act)["params"]
        self.target_critic_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=lr))

        # Actor
        sac_actor = Actor(act_dim)
        self.actor = DrQActor(encoder=encoder,
                              actor=sac_actor,
                              emb_dim=emb_dim)
        actor_params = self.actor.init(actor_key, dummy_obs)["params"]
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
        mean_action, action_dist = self.actor.apply({"params": params}, observation)
        sampled_action, _ = action_dist.sample_and_log_prob(seed=rng)
        return mean_action, sampled_action

    def sample_action(self, observation, eval_mode: bool = False):
        self.rng, sample_rng = jax.random.split(self.rng)
        mean_action, sampled_action = self._sample_action(self.actor_state.params,
                                                          sample_rng,
                                                          observation["pixels"])
        action = mean_action if eval_mode else sampled_action
        action = np.asarray(action)
        return action.clip(-self.max_action, self.max_action)

    def critic_train_step(self,
                          key,
                          alpha_state,
                          actor_state,
                          critic_state,
                          target_critic_params,
                          observations,
                          actions,
                          rewards,
                          next_observations,
                          masks):
        _, dist = self.actor.apply({"params": actor_state.params}, next_observations)
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
        next_qs = self.critic.apply({"params": target_critic_params},
                                    next_observations, next_actions)

        log_alpha = self.log_alpha.apply({"params": alpha_state.params})
        alpha = jnp.exp(log_alpha)
        next_q = next_qs.min(axis=0) - alpha * next_log_probs
        target_q = rewards + self.gamma * masks * next_q

        def critic_loss_fn(critic_params):
            qs = self.critic.apply({"params": critic_params}, observations, actions)
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
                "target_actor_entropy": -next_log_probs.mean(),
            }
        grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=grads)
        new_critic_target_params = target_update(new_critic_state.params,
                                                 target_critic_params,
                                                 self.tau)
        return new_critic_state, new_critic_target_params, info

    def actor_train_step(self,
                         key,
                         alpha_state,
                         actor_state,
                         critic_state,
                         observations):

        def actor_loss_fn(alpha_params, actor_params):
            _, dist = self.actor.apply({"params": actor_params}, observations)
            actions, log_probs = dist.sample_and_log_prob(seed=key)

            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha = jnp.exp(log_alpha)
            alpha_loss = -alpha * jax.lax.stop_gradient(log_probs + self.target_entropy).mean()
            alpha = jax.lax.stop_gradient(alpha)

            qs = self.critic.apply({"params": critic_state.params}, observations, actions)
            q = qs.mean(axis=0)
            actor_loss = (alpha * log_probs - q).mean()
            total_loss = alpha_loss + actor_loss
            return total_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, info = jax.grad(actor_loss_fn, argnums=(0, 1), has_aux=True)(alpha_state.params,
                                                                            actor_state.params)
        alpha_grads, actor_grads = grads
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)

        return new_alpha_state, new_actor_state, info

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   rng,
                   alpha_state,
                   actor_state,
                   critic_state,
                   target_critic_params,
                   batch):

        actor_state = _share_encoder(source=critic_state, target=actor_state)

        rng, aug_key1, aug_key2, actor_key, critic_key = jax.random.split(rng, 5)
        aug_pixels = batched_random_crop(aug_key1, batch.observations[..., :-1])
        aug_next_pixels = batched_random_crop(aug_key2, batch.observations[..., 1:])

        (new_critic_state,
         new_target_critic_params,
         critic_info) = self.critic_train_step(critic_key,
                                               alpha_state,
                                               actor_state,
                                               critic_state,
                                               target_critic_params,
                                               aug_pixels, 
                                               batch.actions,
                                               batch.rewards,
                                               aug_next_pixels,
                                               batch.discounts)

        (new_alpha_state,
         new_actor_state,
         actor_info) = self.actor_train_step(actor_key,
                                             alpha_state,
                                             actor_state,
                                             critic_state,
                                             aug_pixels)

        return rng, new_alpha_state, new_actor_state, new_critic_state, \
            new_target_critic_params, {**critic_info, **actor_info},

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (self.rng,
         self.alpha_state,
         self.actor_state,
         self.critic_state,
         self.target_critic_params,
         info) = self.train_step(self.rng,
                                 self.alpha_state,
                                 self.actor_state,
                                 self.critic_state,
                                 self.target_critic_params,
                                 batch)
        return info
