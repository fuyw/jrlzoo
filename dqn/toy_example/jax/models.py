from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import functools
import jax
import jax.numpy as jnp
import optax
from utils import target_update, Batch


class MLP(nn.Module):
    hidden_dims: Sequence[int] = (64, 64)
    init_fn: Callable = nn.initializers.lecun_normal()
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.init_fn)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


class QNetwork(nn.Module):
    act_dim: int
    hidden_dims: Sequence[int] = (64, 64)
    
    @nn.compact
    def __call__(self, observations: jnp.ndarray):
        Qs = MLP((*self.hidden_dims, self.act_dim), activate_final=False)(observations)
        return Qs


class DQNAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 seed: int = 42,
                 per: bool = False,
                 per_alpha: float = 0.6,
                 hidden_dims: Sequence[int] = (64, 64)):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.per = per
        self.per_alpha = per_alpha

        rng = jax.random.PRNGKey(seed)

        self.net = QNetwork(act_dim, hidden_dims)
        dummy_obs = jnp.ones([1, obs_dim])
        params = self.net.init(rng, dummy_obs)["params"]
        self.target_params = params
        self.state = train_state.TrainState.create(
            apply_fn=QNetwork.apply, params=params, tx=optax.adam(lr))

    @functools.partial(jax.jit, static_argnames=("self"))
    def sample_action(self, params: FrozenDict, observation: jnp.ndarray):
        Qs = self.net.apply({"params": params}, observation) 
        action = Qs.argmax()
        return action

    @functools.partial(jax.jit, static_argnames=("self", "per"))
    def train_step(self,
                   batch: Batch,
                   state: train_state.TrainState,
                   target_params: FrozenDict,
                   per: bool = False):
        next_Q = self.net.apply({"params": target_params}, batch.next_observations).max(-1)
        target_Q = batch.rewards + self.gamma * batch.discounts * next_Q
        def loss_fn(params):
            Qs = self.net.apply({"params": params}, batch.observations)
            Q = jnp.take_along_axis(Qs, jnp.array(batch.actions), -1).squeeze()
            if per:
                loss = (batch.weights * jnp.square(Q - target_Q)).mean()
                td_loss = jnp.abs(Q - target_Q)
                priority = jnp.power(td_loss, self.per_alpha)
                return loss, {"loss": loss, "Q": Q.mean(), "target_Q": target_Q.mean(), "priority": priority}
            else:
                loss = jnp.square(Q - target_Q).mean()
                return loss, {"loss": loss, "Q": Q.mean(), "target_Q": target_Q.mean()}
        (_, log_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        new_target_params = target_update(new_state.params, target_params, self.tau)
        log_info.update({"batch_rewards": batch.rewards.sum(), "batch_actions": batch.actions.sum(),
                         "batch_observations": batch.observations.sum()})
        return new_state, new_target_params, log_info

    def update(self, batch: Batch):
        self.state, self.target_params, log_info = self.train_step(batch, self.state, self.target_params, self.per)
        return log_info


class DDQNAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 seed: int = 42,
                 per: bool = False,
                 per_alpha: float = 0.6,
                 hidden_dims: Sequence[int] = (64, 64)):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.per = per
        self.per_alpha = per_alpha

        rng = jax.random.PRNGKey(seed)

        self.net = QNetwork(act_dim, hidden_dims)
        dummy_obs = jnp.ones([1, obs_dim])
        params = self.net.init(rng, dummy_obs)["params"]
        self.target_params = params
        self.state = train_state.TrainState.create(
            apply_fn=QNetwork.apply, params=params, tx=optax.adam(lr))

    @functools.partial(jax.jit, static_argnames=("self"))
    def sample_action(self, params: FrozenDict, observation: jnp.ndarray):
        Qs = self.net.apply({"params": params}, observation) 
        action = Qs.argmax()
        return action

    @functools.partial(jax.jit, static_argnames=("self", "per"))
    def train_step(self,
                   batch: Batch,
                   state: train_state.TrainState,
                   target_params: FrozenDict,
                   per: bool = False):
        next_Qs = self.net.apply({"params": state.params}, batch.next_observations)
        next_actions = next_Qs.argmax(-1, keepdims=True)
        target_next_Qs = self.net.apply({"params": target_params}, batch.next_observations)
        next_Q = jnp.take_along_axis(target_next_Qs, next_actions, -1).squeeze()
        target_Q = batch.rewards + self.gamma * batch.discounts * next_Q
        def loss_fn(params):
            Qs = self.net.apply({"params": params}, batch.observations)
            Q = jnp.take_along_axis(Qs, jnp.array(batch.actions), -1).squeeze()
            if per:
                loss = (batch.weights * jnp.square(Q - target_Q)).mean()
                td_loss = jnp.abs(Q - target_Q)
                priority = jnp.power(td_loss, self.per_alpha)
                return loss, {"loss": loss, "Q": Q.mean(), "target_Q": target_Q.mean(), "priority": priority}
            else:
                loss = jnp.square(Q - target_Q).mean()
                return loss, {"loss": loss, "Q": Q.mean(), "target_Q": target_Q.mean()}
        (_, log_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        new_target_params = target_update(new_state.params, target_params, self.tau)
        log_info.update({"batch_rewards": batch.rewards.sum(), "batch_actions": batch.actions.sum(),
                         "batch_observations": batch.observations.sum()})
        return new_state, new_target_params, log_info

    def update(self, batch: Batch):
        self.state, self.target_params, log_info = self.train_step(batch, self.state, self.target_params, self.per)
        return log_info
