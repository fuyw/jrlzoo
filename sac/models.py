from typing import Any, Optional
import functools
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils import Batch


LOG_STD_MAX = 2.
LOG_STD_MIN = -10.


class Actor(nn.Module):
    act_dim: int

    def setup(self):
        self.l1 = nn.Dense(256, name="fc1")
        self.l2 = nn.Dense(256, name="fc2")
        self.l3 = nn.Dense(2 * self.act_dim, name="fc3")

    def __call__(self,
                 rng,
                 observation: jnp.ndarray):
        x = nn.relu(self.l1(observation))
        x = nn.relu(self.l2(x))
        x = self.l3(x)
        mu, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
 
        mean_action = jnp.tanh(mu)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1)
        )
        sampled_action, logp = action_distribution.sample_and_log_prob(seed=rng)
        return mean_action, sampled_action, logp


class Critic(nn.Module):
    hid_dim: int = 256

    @nn.compact
    def __call__(self,
                 observation: jnp.ndarray,
                 action: jnp.ndarray) -> jnp.ndarray:
        kernel_initializer = jax.nn.initializers.glorot_uniform()
        x = jnp.concatenate([observation, action], axis=-1)
        q = nn.relu(nn.Dense(self.hid_dim, kernel_init=kernel_initializer,
                             name="fc1")(x))
        q = nn.relu(nn.Dense(self.hid_dim, kernel_init=kernel_initializer,
                             name="fc2")(q))
        q = nn.Dense(1, kernel_init=kernel_initializer, name="fc3")(q)
        return q


class DoubleCritic(nn.Module):
    hid_dim: int = 256

    def setup(self):
        self.critic1 = Critic(self.hid_dim)
        self.critic2 = Critic(self.hid_dim)

    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observation, action)
        q2 = self.critic2(observation, action)
        return q1, q2


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)
    
    def __call__(self):
        return self.value


class SACAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 learning_rate: float = 3e-4,
                 auto_entropy_tuning: bool = True,
                 target_entropy: Optional[float] = None):

        self.update_step = 0
        self.tau = tau
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.auto_entropy_tuning = auto_entropy_tuning
        if target_entropy is None:
            self.target_entropy = -act_dim / 2
        else:
            self.target_entropy = target_entropy

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(act_dim)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.adam(self.learning_rate))

        # Initialize the Critic
        self.critic = DoubleCritic()
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=Critic.apply,
            params=critic_params,
            tx=optax.adam(self.learning_rate))

        # Entropy tuning
        if self.auto_entropy_tuning:
            self.rng, alpha_key = jax.random.split(self.rng, 2)
            self.log_alpha = Scalar(0.0)
            self.alpha_state = train_state.TrainState.create(
                apply_fn=None,
                params=self.log_alpha.init(alpha_key)["params"],
                tx=optax.adam(self.learning_rate)
            )
    
    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   critic_target_params: FrozenDict,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   alpha_state: train_state.TrainState,
                   key: jnp.ndarray):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        observations, actions, rewards, discounts, next_observations = batch
        def loss_fn(actor_params: FrozenDict,
                    critic_params: FrozenDict,
                    alpha_params: FrozenDict,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    discount: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    rng: jnp.ndarray):

            rng1, rng2 = jax.random.split(rng, 2)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply({"params": actor_params},
                                                       rng1, observation)

            # Alpha loss
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + self.target_entropy).mean()
            alpha = jnp.exp(log_alpha)

            # Evaluate sampled actions with Critic
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.squeeze(jnp.minimum(sampled_q1, sampled_q2))

            # Actor loss
            alpha = jax.lax.stop_gradient(alpha)
            actor_loss = (alpha * logp - sampled_q).mean()

            # Critic loss
            q1, q2 = self.critic.apply({"params": critic_params},
                                       observation, action)
            q1 = jnp.squeeze(q1)
            q2 = jnp.squeeze(q2)
            _, next_action, logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng2, next_observation)
            next_q1, next_q2 = self.critic.apply({"params": critic_target_params}, next_observation, next_action)
            next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2)) - \
                alpha * logp_next_action

            target_q = reward + self.gamma * discount * next_q
            target_q = jax.lax.stop_gradient(target_q)
            critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

            # Loss weight form Dopamine
            total_loss = 0.5 * critic_loss + actor_loss + alpha_loss
            log_info = {"q1": q1.mean(), "q2": q2.mean(), "critic_loss": critic_loss, "actor_loss": actor_loss, "alpha_loss": alpha_loss, "alpha": alpha}

            return total_loss, log_info
        
        grad_fn = jax.vmap(
            jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
            in_axes=(None, None, None, 0, 0, 0, 0, 0, 0))
        rng = jnp.stack(jax.random.split(key, num=actions.shape[0]))

        (_, log_info), gradients = grad_fn(actor_state.params,
                                           critic_state.params,
                                           alpha_state.params,
                                           observations,
                                           actions,
                                           rewards,
                                           discounts,
                                           next_observations,
                                           rng)

        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)

        actor_grads, critic_grads, alpha_grads = gradients

        # Update TrainState
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        return log_info, actor_state, critic_state, alpha_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def update_target_params(self, params: FrozenDict, target_params: FrozenDict):
        def _update(param, target_param):
            return self.tau * param + (1 - self.tau) * target_param

        updated_params = jax.tree_multimap(_update, params, target_params)
        return updated_params

    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray, eval_mode: bool = False) -> jnp.ndarray:
        observation = jax.device_put(observation[None])
        rng, sample_rng = jax.random.split(rng)
        mean_action, sampled_action, _ = self.actor.apply({"params": params}, sample_rng, observation)
        return rng, jnp.where(eval_mode, mean_action.flatten(), sampled_action.flatten())

    def update(self, replay_buffer, batch_size: int = 256):
        self.update_step += 1

        # Sample from the buffer
        batch = replay_buffer.sample(batch_size)

        self.rng, key = jax.random.split(self.rng)
        log_info, self.actor_state, self.critic_state, self.alpha_state = self.train_step(batch,
                                                                                          self.critic_target_params,
                                                                                          self.actor_state,
                                                                                          self.critic_state,
                                                                                          self.alpha_state,
                                                                                          key)

        # update target network
        params = self.critic_state.params
        target_params = self.critic_target_params
        self.critic_target_params = self.update_target_params(params, target_params)

        return log_info
