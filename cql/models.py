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
LOG_STD_MIN = -20.

kernel_initializer = jax.nn.initializers.glorot_uniform()


class Actor(nn.Module):
    act_dim: int
    hid_dim: int = 256
    hid_layers: int = 3

    @nn.compact
    def __call__(self, rng: Any, observation: jnp.ndarray):
        x = nn.relu(
            nn.Dense(self.hid_dim, kernel_init=kernel_initializer,
                     name="fc1")(observation))
        for layer_idx in range(1, self.hid_layers):
            x = nn.relu(
                nn.Dense(self.hid_dim,
                         kernel_init=kernel_initializer,
                         name=f"fc{layer_idx+1}")(x))
        x = nn.Dense(2 * self.act_dim,
                     kernel_init=kernel_initializer,
                     name=f"fc{self.hid_layers + 1}")(x)

        mu, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        mean_action = jnp.tanh(mu)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))
        sampled_action, logp = action_distribution.sample_and_log_prob(
            seed=rng)
        return mean_action, sampled_action, logp


class Critic(nn.Module):
    hid_dim: int = 256
    hid_layers: int = 3

    @nn.compact
    def __call__(self, observation: jnp.ndarray,
                 action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observation, action], axis=-1)
        x = nn.relu(
            nn.Dense(self.hid_dim, kernel_init=kernel_initializer,
                     name="fc1")(x))
        for layer_idx in range(1, self.hid_layers):
            x = nn.relu(
                nn.Dense(self.hid_dim,
                         kernel_init=kernel_initializer,
                         name=f"fc{layer_idx+1}")(x))
        q = nn.Dense(1,
                     kernel_init=kernel_initializer,
                     name=f"fc{self.hid_layers+1}")(x)
        return q


class DoubleCritic(nn.Module):
    hid_dim: int = 256
    hid_layers: int = 3

    def setup(self):
        self.critic1 = Critic(self.hid_dim, self.hid_layers)
        self.critic2 = Critic(self.hid_dim, self.hid_layers)

    def __call__(self, observation: jnp.ndarray,
                 action: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observation, action)
        q2 = self.critic2(observation, action)
        return q1, q2


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


class CQLAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hid_dim: int = 256,
                 hid_layers: int = 3,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 lr_actor: float = 3e-4,
                 auto_entropy_tuning: bool = True,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = False,
                 num_random: int = 10,
                 min_q_weight: float = 5.0,
                 with_lagrange: bool = False,
                 subtract_likelihood: bool = True,
                 lagrange_thresh: int = 5.0):

        self.update_step = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim
        self.hid_layers = hid_layers
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.lr_actor = lr_actor
        self.auto_entropy_tuning = auto_entropy_tuning
        self.backup_entropy = backup_entropy
        self.subtract_likelihood = subtract_likelihood
        if target_entropy is None:
            self.target_entropy = -self.act_dim
        else:
            self.target_entropy = target_entropy

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        # Dummy inputs
        dummy_obs = jnp.ones([1, self.obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, self.act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(self.act_dim, self.hid_dim, self.hid_layers)
        actor_params = self.actor.init(actor_key, actor_key,
                                       dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(apply_fn=Actor.apply,
                                                         params=actor_params,
                                                         tx=optax.adam(
                                                             self.lr_actor))

        # Initialize the Critic
        self.critic = DoubleCritic(self.hid_dim, self.hid_layers)
        critic_params = self.critic.init(critic_key, dummy_obs,
                                         dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=DoubleCritic.apply,
            params=critic_params,
            tx=optax.adam(self.lr))

        # Entropy tuning
        if self.auto_entropy_tuning:
            self.rng, alpha_key = jax.random.split(self.rng, 2)
            self.log_alpha = Scalar(0.0)
            self.alpha_state = train_state.TrainState.create(
                apply_fn=None,
                params=self.log_alpha.init(alpha_key)["params"],
                tx=optax.adam(self.lr))

        # CQL parameters
        self.num_random = num_random
        self.with_lagrange = with_lagrange
        self.min_q_weight = min_q_weight if not with_lagrange else 1.0
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.rng, cql_key = jax.random.split(self.rng, 2)
            self.log_cql_alpha = Scalar(0.0)  # 1.0
            self.cql_alpha_state = train_state.TrainState.create(
                apply_fn=None,
                params=self.log_cql_alpha.init(cql_key)["params"],
                tx=optax.adam(self.lr))

    @functools.partial(jax.jit, static_argnames=("self"))
    def update_target_params(self, params: FrozenDict,
                             target_params: FrozenDict):
        def _update(param, target_param):
            return self.tau * param + (1 - self.tau) * target_param

        updated_params = jax.tree_multimap(_update, params, target_params)
        return updated_params

    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action(self,
                      params: FrozenDict,
                      rng: Any,
                      observation: np.ndarray,
                      eval_mode: bool = False) -> jnp.ndarray:
        observation = jax.device_put(observation[None])
        rng, sample_rng = jax.random.split(rng)
        mean_action, sampled_action, _ = self.actor.apply({"params": params},
                                                          sample_rng,
                                                          observation)
        return rng, jnp.where(eval_mode, mean_action.flatten(),
                              sampled_action.flatten())

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self, batch: Batch, critic_target_params: FrozenDict,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   alpha_state: train_state.TrainState, key: jnp.ndarray):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        discounts = batch.discounts
        next_observations = batch.next_observations

        def loss_fn(actor_params: FrozenDict, critic_params: FrozenDict,
                    alpha_params: FrozenDict, observation: jnp.ndarray,
                    action: jnp.ndarray, reward: jnp.ndarray,
                    discount: jnp.ndarray, next_observation: jnp.ndarray,
                    rng: jnp.ndarray):
            """compute loss for a single transition"""
            rng, rng1, rng2 = jax.random.split(rng, 3)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply(
                {"params": actor_params}, rng1, observation)

            # Alpha loss: stop gradient to avoid affect Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(
                logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply(
                {"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.squeeze(jnp.minimum(sampled_q1, sampled_q2))

            # Actor loss
            alpha = jax.lax.stop_gradient(
                alpha)  # stop gradient to avoid affect Alpha parameters
            actor_loss = (alpha * logp - sampled_q)

            # Critic loss
            q1, q2 = self.critic.apply({"params": critic_params}, observation,
                                       action)
            q1 = jnp.squeeze(q1)
            q2 = jnp.squeeze(q2)

            # Use frozen_actor_params to avoid affect Actor parameters
            _, next_action, logp_next_action = self.actor.apply(
                {"params": frozen_actor_params}, rng2, next_observation)
            next_q1, next_q2 = self.critic.apply(
                {"params": critic_target_params}, next_observation,
                next_action)

            next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2))
            if self.backup_entropy:
                next_q -= alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2

            # CQL loss
            rng3, rng4 = jax.random.split(rng, 2)
            cql_random_actions = jax.random.uniform(rng3,
                                                    shape=(self.num_random,
                                                           self.act_dim),
                                                    minval=-1.0,
                                                    maxval=1.0)

            # Sample 10 actions with current state
            repeat_observations = jnp.repeat(jnp.expand_dims(observation,
                                                             axis=0),
                                             repeats=self.num_random,
                                             axis=0)
            repeat_next_observations = jnp.repeat(jnp.expand_dims(
                next_observation, axis=0),
                                                  repeats=self.num_random,
                                                  axis=0)
            _, cql_sampled_actions, cql_logp = self.actor.apply(
                {"params": frozen_actor_params}, rng3, repeat_observations)
            _, cql_next_actions, cql_logp_next_action = self.actor.apply(
                {"params": frozen_actor_params}, rng4,
                repeat_next_observations)

            cql_random_q1, cql_random_q2 = self.critic.apply(
                {"params": critic_params}, repeat_observations,
                cql_random_actions)
            cql_q1, cql_q2 = self.critic.apply({"params": critic_params},
                                               repeat_observations,
                                               cql_sampled_actions)
            cql_next_q1, cql_next_q2 = self.critic.apply(
                {"params": critic_params}, repeat_observations,
                cql_next_actions)

            # Simulate logsumexp() for continuous actions
            random_density = np.log(0.5**self.act_dim)
            cql_random_q1_minus_logp = jnp.squeeze(
                cql_random_q1) - random_density
            cql_next_q1_minus_logp = jnp.squeeze(
                cql_next_q1) - cql_logp_next_action
            cql_q1_minus_logp = jnp.squeeze(cql_q1) - cql_logp

            cql_random_q2_minus_logp = jnp.squeeze(
                cql_random_q2) - random_density
            cql_next_q2_minus_logp = jnp.squeeze(
                cql_next_q2) - cql_logp_next_action
            cql_q2_minus_logp = jnp.squeeze(cql_q2) - cql_logp

            cql_concat_q1 = jnp.concatenate([
                cql_random_q1_minus_logp, cql_next_q1_minus_logp,
                cql_q1_minus_logp
            ])
            cql_concat_q2 = jnp.concatenate([
                cql_random_q2_minus_logp, cql_next_q2_minus_logp,
                cql_q2_minus_logp
            ])

            # CQL0: conservative penalty ==> dominate by the max(cql_concat_q)
            ood_q1 = jax.scipy.special.logsumexp(cql_concat_q1)
            ood_q2 = jax.scipy.special.logsumexp(cql_concat_q2)

            # CQL1: maximize Q(s, a) in the dataset
            if self.subtract_likelihood:
                cql1_loss = (ood_q1 - q1) * self.min_q_weight
                cql2_loss = (ood_q1 - q2) * self.min_q_weight
            else:
                cql1_loss = ood_q1 * self.min_q_weight
                cql2_loss = ood_q2 * self.min_q_weight

            # Loss weight form Dopamine
            total_loss = 0.5 * critic_loss + actor_loss + alpha_loss + cql1_loss + cql2_loss
            log_info = {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "cql1_loss": cql1_loss,
                "cql2_loss": cql2_loss,
                "q1": q1,
                "q2": q2,
                "target_q": target_q,
                "sampled_q": sampled_q.mean(),
                "ood_q1": ood_q1,
                "ood_q2": ood_q2,
                "cql_q1_avg": cql_q1.mean(),
                "cql_q1_min": cql_q1.min(),
                "cql_q1_max": cql_q1.max(),
                "cql_q2_avg": cql_q2.mean(),
                "cql_q2_min": cql_q2.min(),
                "cql_q2_max": cql_q2.max(),
                "cql_concat_q1_avg": cql_concat_q1.mean(),
                "cql_concat_q1_min": cql_concat_q1.min(),
                "cql_concat_q1_max": cql_concat_q1.max(),
                "cql_concat_q2_avg": cql_concat_q2.mean(),
                "cql_concat_q2_min": cql_concat_q2.min(),
                "cql_concat_q2_max": cql_concat_q2.max(),
                "cql_logp": cql_logp.mean(),
                "cql_logp_next_action": cql_logp_next_action.mean(),
                "cql_next_q1_avg": cql_next_q1.mean(),
                "cql_next_q1_min": cql_next_q1.min(),
                "cql_next_q1_max": cql_next_q1.max(),
                "cql_next_q2_avg": cql_next_q2.mean(),
                "cql_next_q2_min": cql_next_q2.min(),
                "cql_next_q2_max": cql_next_q2.max(),
                "random_q1_avg": cql_random_q1.mean(),
                "random_q1_min": cql_random_q1.min(),
                "random_q1_max": cql_random_q1.max(),
                "random_q2_avg": cql_random_q2.mean(),
                "random_q2_min": cql_random_q2.min(),
                "random_q2_max": cql_random_q2.max(),
                "random_q1_IS_avg": cql_random_q1_minus_logp.mean(),
                "random_q1_IS_min": cql_random_q1_minus_logp.min(),
                "random_q1_IS_max": cql_random_q1_minus_logp.max(),
                "next_q1_IS_avg": cql_next_q1_minus_logp.mean(),
                "next_q1_IS_min": cql_next_q1_minus_logp.min(),
                "next_q1_IS_max": cql_next_q1_minus_logp.max(),
                "q1_IS_avg": cql_q1_minus_logp.mean(),
                "q1_IS_min": cql_q1_minus_logp.min(),
                "q1_IS_max": cql_q1_minus_logp.max(),
                "random_q2_IS_avg": cql_random_q2_minus_logp.mean(),
                "random_q2_IS_min": cql_random_q2_minus_logp.min(),
                "random_q2_IS_max": cql_random_q2_minus_logp.max(),
                "next_q2_IS_avg": cql_next_q2_minus_logp.mean(),
                "next_q2_IS_min": cql_next_q2_minus_logp.min(),
                "next_q2_IS_max": cql_next_q2_minus_logp.max(),
                "q2_IS_avg": cql_q2_minus_logp.mean(),
                "q2_IS_min": cql_q2_minus_logp.min(),
                "q2_IS_max": cql_q2_minus_logp.max(),
                "alpha": alpha,
                "logp": logp,
                "logp_next_action": logp_next_action
            }

            return total_loss, log_info

        grad_fn = jax.vmap(jax.value_and_grad(loss_fn,
                                              argnums=(0, 1, 2),
                                              has_aux=True),
                           in_axes=(None, None, None, 0, 0, 0, 0, 0, 0))
        rng = jnp.stack(jax.random.split(key, num=actions.shape[0]))

        (_, log_info), gradients = grad_fn(actor_state.params,
                                           critic_state.params,
                                           alpha_state.params, observations,
                                           actions, rewards, discounts,
                                           next_observations, rng)

        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0),
                                 gradients)
        extra_log_info = {
            'q1_min': log_info['q1'].min(),
            'q1_max': log_info['q1'].max(),
            'q1_std': log_info['q1'].std(),
            'q2_min': log_info['q2'].min(),
            'q2_max': log_info['q2'].max(),
            'q2_std': log_info['q2'].std(),
            'target_q_min': log_info['target_q'].min(),
            'target_q_max': log_info['target_q'].max(),
            'target_q_std': log_info['target_q'].std(),
            'ood_q1_min': log_info['ood_q1'].min(),
            'ood_q1_max': log_info['ood_q1'].max(),
            'ood_q1_std': log_info['ood_q1'].std(),
            'ood_q2_min': log_info['ood_q2'].min(),
            'ood_q2_max': log_info['ood_q2'].max(),
            'ood_q2_std': log_info['ood_q2'].std(),
            'critic_loss_min': log_info['critic_loss'].min(),
            'critic_loss_max': log_info['critic_loss'].max(),
            'critic_loss_std': log_info['critic_loss'].std(),
            'cql1_loss_min': log_info['cql1_loss'].min(),
            'cql1_loss_max': log_info['cql1_loss'].max(),
            'cql1_loss_std': log_info['cql1_loss'].std(),
            'cql2_loss_min': log_info['cql2_loss'].min(),
            'cql2_loss_max': log_info['cql2_loss'].max(),
            'cql2_loss_std': log_info['cql2_loss'].std(),
        }
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        log_info.update(extra_log_info)
        actor_grads, critic_grads, alpha_grads = gradients

        # Update TrainState
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        return log_info, actor_state, critic_state, alpha_state

    def update(self, replay_buffer, batch_size: int = 256):
        self.update_step += 1

        # Sample from the buffer
        batch = replay_buffer.sample(batch_size)
        self.rng, key = jax.random.split(self.rng)
        (log_info, self.actor_state, self.critic_state,
         self.alpha_state) = self.train_step(batch, self.critic_target_params,
                                             self.actor_state,
                                             self.critic_state,
                                             self.alpha_state, key)
        log_info['batch_rewards'] = batch.rewards.mean().item()
        log_info['batch_discounts'] = batch.discounts.mean().item()
        log_info['batch_obs'] = abs(batch.observations).sum(1).mean().item()

        # update target network
        params = self.critic_state.params
        target_params = self.critic_target_params
        self.critic_target_params = self.update_target_params(
            params, target_params)

        return log_info
