import os
import functools
from typing import Any, Callable, Dict, Optional, Sequence

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import orbax_utils, train_state

from utils import Batch, target_update


###################
# Utils Functions #
###################
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


################
# Actor-Critic #
################
class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    min_scale: float = 1e-3

    def setup(self):
        self.net = MLP(self.hidden_dims, activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim)
        self.std_layer = nn.Dense(self.act_dim)

    def __call__(self, rng: Any, observation: jnp.ndarray, goal: jnp.ndarray):
        x = jnp.concatenate([observation, goal], axis=-1)
        x = self.net(x)
        mu = self.mu_layer(x)
        mean_action = nn.tanh(mu)

        std = self.std_layer(x)
        std = jax.nn.softplus(std) + self.min_scale

        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))
        sampled_action, logp = action_distribution.sample_and_log_prob(
            seed=rng)

        return mean_action * self.max_action, sampled_action * self.max_action, logp

    def get_logprob(self, observation, goal, action):
        x = jnp.concatenate([observation, goal], axis=-1)
        x = self.net(x)
        mu = self.mu_layer(x)
        mean_action = nn.tanh(mu)

        std = self.std_layer(x)
        std = jax.nn.softplus(std) + self.min_scale

        action_distribution = distrax.Normal(mu, std)
        raw_action = atanh(action)
        log_prob = action_distribution.log_prob(raw_action).sum(-1)
        log_prob -= 2 * (jnp.log(2) - raw_action -
                         jax.nn.softplus(-2 * raw_action)).sum(-1)
        return log_prob, mu, std


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    output_dim: int = 1

    def setup(self):
        self.net = MLP(self.hidden_dims, activate_final=True)
        self.out_layer = nn.Dense(self.output_dim)

    def __call__(self,
                 observation: jnp.ndarray,
                 goal: jnp.ndarray,
                 action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observation, goal, action], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze()


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    output_dim: int = 1
    num_qs: int = 2

    @nn.compact
    def __call__(self, observation, goal, action):
        VmapCritic = nn.vmap(Critic,
                             variable_axes={"params": 0},
                             split_rngs={"params": True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims, self.output_dim)(observation,
                                                           goal,
                                                           action)
        return qs


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


#############
# SAC Agent #
#############
class SACAgent:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 goal_dim: int,
                 max_action: float = 1.0,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 ckpt_dir: str = None):

        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.act_dim = act_dim
        self.max_action = max_action
        self.target_entropy = -act_dim / 2

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_goal = jnp.ones([1, goal_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Create the optimizer
        actor_tx = optax.adam(lr)
        critic_tx = optax.adam(lr)

        # Initialize the Actor
        self.actor = Actor(act_dim=act_dim,
                           max_action=max_action,
                           hidden_dims=hidden_dims)
        actor_params = self.actor.init(actor_key,
                                       actor_key,
                                       dummy_obs,
                                       dummy_goal)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply, params=actor_params, tx=actor_tx)

        # Initialize the Critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims)
        critic_params = self.critic.init(critic_key,
                                         dummy_obs,
                                         dummy_goal,
                                         dummy_act,)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply, params=critic_params, tx=critic_tx)

        # Entropy tuning
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.adam(lr))

        # Checkpoint
        self.ckpt_dir = ckpt_dir
        self.checkpointer = ocp.PyTreeCheckpointer()

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self,
                       params: FrozenDict,
                       rng: Any,
                       observation: np.ndarray,
                       goal: np.ndarray) -> jnp.ndarray:
        mean_action, sampled_action, _ = self.actor.apply({"params": params},
                                                          rng,
                                                          observation,
                                                          goal)
        return mean_action, sampled_action

    def sample_action(self,
                      observation: np.ndarray,
                      goal: np.ndarray,
                      eval_mode: bool = False) -> np.ndarray:
        # for deterministic result
        if eval_mode:
            sample_rng = self.rng
        else:
            self.rng, sample_rng = jax.random.split(self.rng)
        mean_action, sampled_action = self._sample_action(
            self.actor_state.params, sample_rng, observation, goal)
        action = mean_action if eval_mode else sampled_action
        action = np.asarray(action)
        return action.clip(-self.max_action, self.max_action)

    def actor_alpha_train_step(self,
                               batch: Batch,
                               key: Any,
                               alpha_state: train_state.TrainState,
                               actor_state: train_state.TrainState,
                               critic_state: train_state.TrainState):

        frozen_critic_params = critic_state.params

        def loss_fn(alpha_params: FrozenDict,
                    actor_params: FrozenDict,
                    rng: Any,
                    observation: jnp.ndarray,
                    goal: jnp.ndarray):
            # sample action with actor
            _, sampled_action, logp = self.actor.apply({"params": actor_params},
                                                       rng,
                                                       observation,
                                                       goal)

            # compute alpha loss
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha = jnp.exp(log_alpha)
            alpha_loss = -alpha * jax.lax.stop_gradient(logp + self.target_entropy)

            # stop alpha gradient
            alpha = jax.lax.stop_gradient(alpha)

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params},
                                                       observation,
                                                       goal,
                                                       sampled_action)
            sampled_q = jnp.minimum(sampled_q1, sampled_q2)

            # Actor loss
            actor_loss = alpha * logp - sampled_q

            # return info
            actor_alpha_loss = actor_loss + alpha_loss
            log_info = {
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "alpha": alpha,
                "logp": logp
            }
            return actor_alpha_loss, log_info

        # compute gradient with vmap
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn,
                                              argnums=(0, 1),
                                              has_aux=True),
                           in_axes=(None, None, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))

        (_, log_info), grads = grad_fn(alpha_state.params,
                                       actor_state.params,
                                       keys,
                                       batch.observations,
                                       batch.goals)
        grads = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), grads)
        log_info = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), log_info)

        # Update TrainState
        alpha_grads, actor_grads = grads
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        return new_alpha_state, new_actor_state, log_info

    def critic_train_step(self,
                          batch: Batch,
                          key: Any,
                          alpha: float,
                          actor_state: train_state.TrainState,
                          critic_state: train_state.TrainState,
                          critic_target_params: FrozenDict):

        frozen_actor_params = actor_state.params

        def loss_fn(params: FrozenDict,
                    rng: Any,
                    observation: jnp.ndarray,
                    goal: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    discount: jnp.ndarray):

            # current q value
            q1, q2 = self.critic.apply({"params": params},
                                       observation,
                                       goal,
                                       action)

            # next q value
            _, next_action, logp_next_action = self.actor.apply({"params": frozen_actor_params},
                                                                rng,
                                                                next_observation,
                                                                goal)
            next_q1, next_q2 = self.critic.apply({"params": critic_target_params},
                                                 next_observation,
                                                 goal,
                                                 next_action)
            next_q = jnp.minimum(next_q1, next_q2) - alpha * logp_next_action

            # target q value
            target_q = reward + self.gamma * discount * next_q

            # td error
            critic_loss1 = (q1 - target_q)**2
            critic_loss2 = (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2
            log_info = {"critic_loss": critic_loss, "q1": q1}
            return critic_loss, log_info

        # compute gradient with vmap
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True),
                           in_axes=(None, 0, 0, 0, 0, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))

        (_, log_info), grads = grad_fn(critic_state.params,
                                       keys,
                                       batch.observations,
                                       batch.goals,
                                       batch.actions,
                                       batch.rewards,
                                       batch.next_observations,
                                       batch.discounts)
        extra_log_info = {"max_q1": log_info["q1"].max(),
                          "min_q1": log_info["q1"].min()}
        grads = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), grads)
        log_info = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        log_info.update(extra_log_info)

        # Update TrainState
        new_critic_state = critic_state.apply_gradients(grads=grads)
        new_critic_target_params = target_update(new_critic_state.params,
                                                 critic_target_params,
                                                 self.tau)
        return new_critic_state, new_critic_target_params, log_info

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   key: Any,
                   alpha_state: train_state.TrainState,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):
        key1, key2 = jax.random.split(key)
        new_alpha_state, new_actor_state, actor_log_info = self.actor_alpha_train_step(
            batch, key1, alpha_state, actor_state, critic_state)
        alpha = actor_log_info["alpha"]
        new_critic_state, new_critic_target_params, critic_log_info = self.critic_train_step(
            batch, key2, alpha, actor_state, critic_state, critic_target_params)
        log_info = {**actor_log_info, **critic_log_info}
        return new_alpha_state, new_actor_state, new_critic_state, new_critic_target_params, log_info

    def update(self, batch: Batch):
        self.rng, key = jax.random.split(self.rng, 2)
        (self.alpha_state,
         self.actor_state,
         self.critic_state,
         self.critic_target_params,
         log_info) = self.train_step(batch,
                                     key,
                                     self.alpha_state,
                                     self.actor_state,
                                     self.critic_state,
                                     self.critic_target_params)
        return log_info

    def save(self, cnt: int = 0):
        params = {"actor": self.actor_state.params,
                  "critic": self.critic_state.params}
        save_args = orbax_utils.save_args_from_target(params)
        self.checkpointer.save(f"{self.ckpt_dir}/{cnt}",
                               params,
                               force=True,
                               save_args=save_args)

    def load(self, ckpt_dir: str, cnt: int = 0):
        raw_restored = self.checkpointer.restore(f"{ckpt_dir}/{cnt}")
        actor_params = raw_restored["actor"]
        critic_params = raw_restored["critic"]

        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=optax.adam(self.lr)) 
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(self.lr))
        self.critic_target_params = critic_params
