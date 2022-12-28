import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import distrax
import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
from utils import target_update, Batch

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


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "glorot_uniform"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.out_layer = nn.Dense(1,
                                  kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze(-1)

    def encode(self, observations: jnp.ndarray,
               actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    # initializer: str = "glorot_uniform"
    initializer: str = "orthogonal"

    def setup(self):
        self.critic1 = Critic(self.hidden_dims, initializer=self.initializer)
        self.critic2 = Critic(self.hidden_dims, initializer=self.initializer)

    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2

    def Q1(self, observations: jnp.ndarray,
           actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observations, actions)
        return q1


class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    # initializer: str = "glorot_uniform"
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

    def __call__(self, rng: Any, observation: jnp.ndarray):
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        mean_action = nn.tanh(mu)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))
        sampled_action, logp = action_distribution.sample_and_log_prob(seed=rng)
        return mean_action * self.max_action, sampled_action * self.max_action, logp

    def get_logp(self, observation: jnp.ndarray,
                 action: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        action_distribution = distrax.Normal(mu, std)
        raw_action = atanh(action)
        logp = action_distribution.log_prob(raw_action).sum(-1)
        logp -= 2 * (jnp.log(2) - raw_action -
                     jax.nn.softplus(-2 * raw_action)).sum(-1)
        return logp



rng = jax.random.PRNGKey(0)
rng, actor_key, critic_key = jax.random.split(rng, 3)

obs_dim, act_dim = 17, 6
dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

actor = Actor(act_dim=act_dim) 
actor_params = actor.init(actor_key, actor_key, dummy_obs)["params"]
actor_state = train_state.TrainState.create(apply_fn=Actor.apply,
                                            params=actor_params,
                                            tx=optax.adam(1e-3))

# Initialize the Critic
critic = DoubleCritic()
critic_params = critic.init(critic_key, dummy_obs, dummy_act)["params"]
critic_target_params = critic_params
critic_state = train_state.TrainState.create(
    apply_fn=Critic.apply, params=critic_params, tx=optax.adam(1e-3))


observations = np.random.normal(size=(16, obs_dim))
next_observations = np.random.normal(size=(16, obs_dim))
actions = np.random.normal(size=(16, act_dim))
rewards = np.random.normal(size=(16,))
discounts = np.random.normal(size=(16,))

frozen_actor_params = actor_state.params


def loss_fn(critic_params: FrozenDict, 
            rng: Any,
            observation: jnp.ndarray,
            action: jnp.ndarray,
            reward: jnp.ndarray,
            next_observation: jnp.ndarray,
            discount: jnp.ndarray):
    """compute loss for a single transition"""
    # Critic loss
    q1, q2 = critic.apply({"params": critic_params}, observation, action)

    # Use frozen_actor_params to avoid affect Actor parameters
    _, next_action, logp_next_action = actor.apply(
        {"params": frozen_actor_params}, rng, next_observation)
    next_q1, next_q2 = critic.apply(
        {"params": critic_target_params}, next_observation, next_action)
    next_q = jnp.minimum(next_q1, next_q2)
    target_q = reward + 0.99 * discount * next_q
    critic_loss1 = (q1 - target_q)**2
    critic_loss2 = (q2 - target_q)**2
    critic_loss = critic_loss1 + critic_loss2

    # Total loss
    log_info = {
        "critic_loss1": critic_loss1,
        "critic_loss2": critic_loss2,
        "critic_loss": critic_loss,
        "q1": q1,
        "q2": q2,
        "target_q": target_q,
    }
    return critic_loss, log_info


keys = jnp.stack(jax.random.split(rng, num=16))
grad_fn = jax.vmap(jax.value_and_grad(loss_fn, argnums=(0), has_aux=True),
                   in_axes=(None, None, 0, 0, 0, 0, 0))
(_, critic_info), critic_grads = grad_fn(critic_state.params,
                                         keys,
                                         observations,
                                         actions,
                                         rewards,
                                         next_observations,
                                         discounts)