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
from load_tf_model import load_model

LOG_STD_MAX = 2.
LOG_STD_MIN = -10.

kernel_initializer = jax.nn.initializers.glorot_uniform()


frozen_actor_params = agent.actor_state.params
frozen_critic_params = agent.critic_state.params



def loss_fn(actor_params: FrozenDict,
            critic_params: FrozenDict,
            alpha_params: FrozenDict,
            observation: jnp.ndarray,
            action: jnp.ndarray,
            reward: jnp.ndarray,
            discount: jnp.ndarray,
            next_observation: jnp.ndarray,
            mask: jnp.ndarray,
            rng: jnp.ndarray):
    """compute loss for a single transition"""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    # Sample actions with Actor
    _, sampled_action, logp = agent.actor.apply(
        {"params": actor_params}, rng1, observation)

    # Alpha loss: stop gradient to avoid affect Actor parameters
    log_alpha = agent.log_alpha.apply({"params": alpha_params})
    alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + agent.target_entropy)
    alpha = jnp.exp(log_alpha)
    alpha = jax.lax.stop_gradient(alpha) 

    # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
    sampled_q1, sampled_q2 = agent.critic.apply(
        {"params": frozen_critic_params}, observation, sampled_action)
    sampled_q = jnp.squeeze(jnp.minimum(sampled_q1, sampled_q2))

    # Actor loss
    # stop gradient to avoid affect Alpha parameters
    actor_loss = (alpha * logp - sampled_q)

    # Critic loss
    q1, q2 = agent.critic.apply({"params": critic_params}, observation, action)
    q1 = jnp.squeeze(q1)
    q2 = jnp.squeeze(q2)

    # Use frozen_actor_params to avoid affect Actor parameters
    _, next_action, logp_next_action = agent.actor.apply(
        {"params": frozen_actor_params}, rng2, next_observation)
    next_q1, next_q2 = agent.critic.apply(
        {"params": agent.critic_target_params}, next_observation,
        next_action)
    next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2))
    if agent.backup_entropy: next_q -= alpha * logp_next_action
    target_q = reward + agent.gamma * discount * next_q
    critic_loss1 = 0.5 * (q1 - target_q)**2
    critic_loss2 = 0.5 * (q2 - target_q)**2
    critic_loss = (critic_loss1 + critic_loss2) * mask * agent.real_loss_ratio

    # CQL loss
    rng3, rng4 = jax.random.split(rng, 2)
    cql_random_actions = jax.random.uniform(
        rng3, shape=(agent.num_random, agent.act_dim), minval=-1.0, maxval=1.0)

    # Sample 10 actions with current state
    repeat_observations = jnp.repeat(jnp.expand_dims(observation,
                                                        axis=0),
                                        repeats=agent.num_random,
                                        axis=0)
    repeat_next_observations = jnp.repeat(jnp.expand_dims(
        next_observation, axis=0),
                                            repeats=agent.num_random,
                                            axis=0)
    _, cql_sampled_actions, cql_logp = agent.actor.apply(
        {"params": frozen_actor_params}, rng3, repeat_observations)
    _, cql_next_actions, cql_logp_next_action = agent.actor.apply(
        {"params": frozen_actor_params}, rng4,
        repeat_next_observations)

    cql_random_q1, cql_random_q2 = agent.critic.apply({"params": critic_params},
                                                        repeat_observations,
                                                        cql_random_actions)
    cql_q1, cql_q2 = agent.critic.apply({"params": critic_params},
                                        repeat_observations,
                                        cql_sampled_actions)
    cql_next_q1, cql_next_q2 = agent.critic.apply(
        {"params": critic_params}, repeat_observations,
        cql_next_actions)

    # Simulate logsumexp() for continuous actions
    random_density = np.log(0.5**agent.act_dim)
    cql_concat_q1 = jnp.concatenate([
        jnp.squeeze(cql_random_q1) - random_density,
        jnp.squeeze(cql_next_q1) - cql_logp_next_action,
        jnp.squeeze(cql_q1) - cql_logp,
    ])
    cql_concat_q2 = jnp.concatenate([
        jnp.squeeze(cql_random_q2) - random_density,
        jnp.squeeze(cql_next_q2) - cql_logp_next_action,
        jnp.squeeze(cql_q2) - cql_logp,
    ])

    # CQL0: conservative penalty ==> dominate by the max(cql_concat_q)
    ood_q1 = jax.scipy.special.logsumexp(cql_concat_q1)
    ood_q2 = jax.scipy.special.logsumexp(cql_concat_q2)

    # CQL1: maximize Q(s, a) in the dataset
    # cql1_loss = agent.min_q_weight * (
    #     ood_q1 - q1) * mask * agent.real_loss_ratio
    # cql2_loss = agent.min_q_weight * (
    #     ood_q2 - q2) * mask * agent.real_loss_ratio

    cql1_loss = agent.min_q_weight * (
        ood_q1 - q1) * mask * agent.real_loss_ratio
    cql2_loss = agent.min_q_weight * (
        ood_q2 - q2) * mask * agent.real_loss_ratio

    # cql1_loss = agent.min_q_weight * (
    #     ood_q1 * (1-mask) * agent.fake_loss_ratio - q1 * mask*agent.real_loss_ratio)
    # cql2_loss = agent.min_q_weight * (
    #     ood_q2 * (1-mask) * agent.fake_loss_ratio - q2 * mask*agent.real_loss_ratio)

    # Loss weight form Dopamine
    total_loss = critic_loss + actor_loss * mask * agent.real_loss_ratio  + alpha_loss * mask * agent.real_loss_ratio + (cql1_loss + cql2_loss)
    log_info = {
        "critic_loss1": critic_loss1,
        "critic_loss2": critic_loss2,
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "cql1_loss": cql1_loss,
        "cql2_loss": cql2_loss,
        "q1": q1,
        "q2": q2,
        "target_q": target_q,
        "sampled_q": sampled_q,
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
        "alpha": alpha,
        "logp": logp,
        "logp_next_action": logp_next_action
    }

    return total_loss, log_info


grad_fn = jax.vmap(jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
                   in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0))
rng = jnp.stack(jax.random.split(jax.random.PRNGKey(10), num=actions.shape[0]))

grad_fn = jax.vmap(jax.value_and_grad(loss_fn,
                                      argnums=(0, 1, 2),
                                      has_aux=True),
                    in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0))

rng = jnp.stack(jax.random.split(jax.random.PRNGKey(2), num=concat_actions.shape[0]))
(_, log_info), gradients = grad_fn(agent.actor_state.params,
                                   agent.critic_state.params,
                                   agent.alpha_state.params,
                                   concat_observations,
                                   concat_actions,
                                   concat_rewards,
                                   concat_discounts,
                                   concat_next_observations,
                                   agent.masks,
                                   rng)
