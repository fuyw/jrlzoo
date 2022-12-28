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

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


def init_fn(initializer: str, gain: float = jnp.sqrt(2)):
    if initializer == "orthogonal":
        return nn.initializers.orthogonal(gain)
    elif initializer == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    elif initializer == "glorot_normal":
        return nn.initializers.glorot_normal()
    return nn.initializers.lecun_normal()


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


class Actor(nn.Module):
    act_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature: float = 1.0):
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)
        means = nn.Dense(self.act_dim)(outputs)
        means = nn.tanh(means)

        log_stds = self.param('log_stds', nn.initializers.zeros, (self.act_dim))
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        stds = jnp.exp(log_stds)

        base_dist = distrax.MultivariateNormalDiag(means, stds*temperature)
        return base_dist


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "glorot_uniform"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(self.initializer), activate_final=True)
        self.out_layer = nn.Dense(1, kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze(-1)

    def encode(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "glorot_uniform"

    def setup(self):
        self.critic1 = Critic(self.hidden_dims, initializer=self.initializer)
        self.critic2 = Critic(self.hidden_dims, initializer=self.initializer)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2

    def Q1(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observations, actions)
        return q1


class AWACAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float = 1.0,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 seed: int = 42,
                 hidden_dims: Sequence[int] = (256, 256),
                 initializer: str = "glorot_uniform"):

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.temperature = 0.5

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        self.actor = Actor(act_dim) 
        actor_params = self.actor.init(actor_key, dummy_obs, 1.0)["params"]
        self.actor_state = train_state.TrainState.create(apply_fn=Actor.apply,
                                                         params=actor_params,
                                                         tx=optax.adam(learning_rate=lr))

        # Initialize the critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(apply_fn=DoubleCritic.apply,
                                                          params=critic_params,
                                                          tx=optax.adam(learning_rate=lr))

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self, params, key, observation, temperature):
        dist = self.actor.apply({"params": params}, observation, temperature)
        sampled_action = dist.sample(seed=key)
        return sampled_action

    def sample_action(self, observation, temperature):
        self.rng, sample_rng = jax.random.split(self.rng, 2)
        sampled_action = self._sample_action(self.actor_state.params, sample_rng, observation, temperature)
        sampled_action = np.asarray(sampled_action)
        return sampled_action.clip(-1.0, 1.0)

    def actor_train_step(self,
                         batch: Batch,
                         actor_key: Any,
                         actor_state: train_state.TrainState,
                         critic_params: FrozenDict):
        q1, q2 = self.critic.apply({"params": critic_params}, batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)

        def loss_fn(actor_params: FrozenDict):
            dist = self.actor.apply({"params": actor_params}, batch.observations, 1.0)
            sampled_actions = dist.sample(seed=actor_key)
            sampled_q1, sampled_q2 = self.critic.apply(
                {"params": critic_params}, batch.observations, jax.lax.stop_gradient(sampled_actions))
            v = jnp.minimum(sampled_q1, sampled_q2)
            logp = dist.log_prob(batch.actions)
            actor_loss = -jax.nn.softmax((q - v)/2.0) * logp

            return actor_loss.mean(), {
                "actor_loss": actor_loss.mean(),
                "actor_loss_max": actor_loss.max(),
                "actor_loss_min": actor_loss.min(),
                "v": v.mean(),
                "v_max": v.max(),
                "v_min": v.min(),
                "logp": logp.mean(),
                "logp_max": logp.max(),
                "logp_min": logp.min(),
            }
        (_, actor_info), actor_grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        return actor_info, actor_state

    def critic_train_step(self,
                          batch: Batch,
                          critic_key: Any,
                          critic_state: train_state.TrainState,
                          actor_params: FrozenDict,
                          critic_target_params: FrozenDict):

        dist = self.actor.apply({"params": actor_params}, batch.next_observations)
        next_actions = dist.sample(seed=critic_key)
        next_q1, next_q2 = self.critic.apply(
            {"params": critic_target_params}, batch.next_observations, next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        target_q = batch.rewards + self.gamma * batch.discounts * next_q

        def loss_fn(params: FrozenDict):
            q1, q2 = self.critic.apply({"params": params}, batch.observations, batch.actions)
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2
            return critic_loss.mean(), {
                "critic_loss": critic_loss.mean(), "critic_loss_max": critic_loss.max(), "critic_loss_min": critic_loss.min(),
                "q1": q1.mean(), "q1_max": q1.max(), "q1_min": q1.min(),
            }
        (_, critic_info), critic_grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   actor_key: Any,
                   critic_key: Any,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   actor_params: FrozenDict,
                   critic_target_params: FrozenDict):

        critic_info, new_critic_state = self.critic_train_step(
            batch, critic_key, critic_state, actor_params, critic_target_params)
        new_critic_target_params = target_update(new_critic_state.params, critic_target_params, self.tau)

        actor_info, new_actor_state = self.actor_train_step(
            batch, actor_key, actor_state, new_critic_state.params)
        return new_actor_state, new_critic_state, new_critic_target_params, {**actor_info, **critic_info}

        # actor_info, new_actor_state = self.actor_train_step(
        #     batch, actor_key, actor_state, critic_state.params)
        # return new_actor_state, critic_state, critic_target_params, actor_info

    def update(self, batch: Batch):
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)
        self.actor_state, self.critic_state, self.critic_target_params, log_info = self.train_step(
            batch, actor_key, critic_key, self.actor_state,
            self.critic_state, self.actor_state.params, self.critic_target_params)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)
