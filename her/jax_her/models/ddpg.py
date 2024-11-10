import functools
import os
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state

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
# Actor Critic #
################
class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.out_layer = nn.Dense(self.act_dim,
                                  kernel_init=init_fn(self.initializer, 1e-2))

    def __call__(self, observation: jnp.ndarray, goal: jnp.ndarray):
        x = jnp.concatenate([observation, goal], axis=-1)
        x = self.net(x)
        x = self.out_layer(x)
        mean_action = nn.tanh(x) * self.max_action
        return mean_action


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(self.initializer), activate_final=True)
        self.out_layer = nn.Dense(1, kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self,
                 observation: jnp.ndarray,
                 goal: jnp.ndarray,
                 action: jnp.ndarray):
        x = jnp.concatenate([observation, goal, action], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze(-1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"
    num_qs: int = 2

    @nn.compact
    def __call__(self, observation, goal, action):
        q1 = Critic(self.hidden_dims, self.initializer)(observation,
                                                        goal,
                                                        action)
        q2 = Critic(self.hidden_dims, self.initializer)(observation,
                                                        goal,
                                                        action)
        return q1, q2


##############
# DDPG Agent #
##############
class DDPGAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 goal_dim: int,
                 max_action: float = 1.0,
                 tau: float = 0.005,
                 gamma: float = 0.99, 
                 lr: float = 3e-4,
                 seed: int = 42,
                 hidden_dims: Sequence[int] = (256, 256),
                 initializer: str = "glorot_uniform",
                 ckpt_dir: str = None):

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        rng = jax.random.PRNGKey(seed)
        self.actor_rng, self.critic_rng = jax.random.split(rng, 2)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)
        dummy_goal = jnp.ones([1, goal_dim], dtype=jnp.float32)

        self.actor = Actor(act_dim=act_dim,
                           max_action=max_action,
                           hidden_dims=hidden_dims,
                           initializer=initializer)
        actor_params = self.actor.init(self.actor_rng,
                                       dummy_obs,
                                       dummy_goal)["params"]
        self.actor_target_params = actor_params
        self.actor_state = train_state.TrainState.create(apply_fn=Actor.apply,
                                                         params=actor_params,
                                                         tx=optax.adam(learning_rate=lr))

        # Initialize the critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(self.critic_rng,
                                         dummy_obs,
                                         dummy_goal,
                                         dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(apply_fn=Critic.apply,
                                                          params=critic_params,
                                                          tx=optax.adam(learning_rate=lr))

        # Checkpoint
        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
            self.checkpointer = ocp.StandardCheckpointer()

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self,
                       params: FrozenDict,
                       observation: np.ndarray,
                       goal: np.ndarray):
        sampled_action = self.actor.apply({"params": params},
                                          observation,
                                          goal)
        return sampled_action

    def sample_action(self, observation: np.ndarray, goal: np.ndarray):
        sampled_action = self._sample_action(self.actor_state.params,
                                             observation,
                                             goal)
        sampled_action = np.asarray(sampled_action)
        return sampled_action

    def actor_train_step(self,
                         batch: Batch,
                         actor_state: train_state.TrainState,
                         critic_params: FrozenDict):
        def loss_fn(params: FrozenDict):
            actions = self.actor.apply({"params": params},
                                       batch.observations,
                                       batch.goals)
            q, _ = self.critic.apply({"params": critic_params},
                                     batch.observations,
                                     batch.goals,
                                     actions)
            actor_loss = -q
            avg_actor_loss = actor_loss.mean()
            return avg_actor_loss, {
                "actor_loss": avg_actor_loss, 
            }
        (_, actor_info), actor_grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        return actor_info, actor_state

    def critic_train_step(self,
                          batch: Batch,
                          critic_key: Any,
                          critic_state: train_state.TrainState,
                          actor_target_params: FrozenDict,
                          critic_target_params: FrozenDict):
        next_actions = self.actor.apply({"params": actor_target_params},
                                        batch.next_observations,
                                        batch.goals)
        next_q1, next_q2 = self.critic.apply({"params": critic_target_params},
                                             batch.next_observations,
                                             batch.goals,
                                             next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        target_q = batch.rewards + self.gamma * batch.discounts * next_q
        def loss_fn(params: FrozenDict):
            q1, q2 = self.critic.apply({"params": params},
                                       batch.observations,
                                       batch.goals,
                                       batch.actions)
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2
            avg_critic_loss = critic_loss.mean()

            return avg_critic_loss, {
                "critic_loss": avg_critic_loss, 
                "q1": q1.mean(),
                "max_q1": q1.max(),
                "min_q1": q1.min(),
            }
        (_, critic_info), critic_grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   actor_target_params: FrozenDict,
                   critic_target_params: FrozenDict,
                   critic_key: Any):
        critic_info, new_critic_state = self.critic_train_step(batch,
                                                               critic_key,
                                                               critic_state, actor_target_params,
                                                               critic_target_params)
        actor_info, new_actor_state = self.actor_train_step(batch,
                                                            actor_state,
                                                            critic_state.params)
        params = (new_actor_state.params, new_critic_state.params)
        target_params = (actor_target_params, critic_target_params)
        new_actor_target_params, new_critic_target_params = target_update(params,
                                                                          target_params,
                                                                          self.tau)
        return new_actor_state, new_critic_state, new_actor_target_params, \
            new_critic_target_params, {**actor_info, **critic_info}
    
    def update(self, batch: Batch):
        self.critic_rng, critic_key = jax.random.split(self.critic_rng, 2) 
        (self.actor_state,
         self.critic_state,
         self.actor_target_params,
         self.critic_target_params,
         log_info) = self.train_step(batch,
                                     self.actor_state,
                                     self.critic_state,
                                     self.actor_target_params,
                                     self.critic_target_params,
                                     critic_key) 
        return log_info

    def save(self, cnt: int = 0):
        params = {"actor": self.actor_state.params,
                  "critic": self.critic_state.params}
        self.checkpointer.save(f"{self.ckpt_dir}/{cnt}",
                               params, force=True)

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
