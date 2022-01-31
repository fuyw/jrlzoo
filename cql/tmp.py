import jax
import jax.numpy as jnp
from flax.core import FrozenDict

batch = replay_buffer.sample(128)
frozen_actor_params = agent.actor_state.params
frozen_critic_params = agent.critic_state.params
observations = batch.observations
actions = batch.actions
rewards = batch.rewards
discounts = batch.discounts
next_observations = batch.next_observations

critic_target_params = agent.critic_target_params


def loss_fn(actor_params: FrozenDict, critic_params: FrozenDict,
            alpha_params: FrozenDict, observation: jnp.ndarray,
            action: jnp.ndarray, reward: jnp.ndarray, discount: jnp.ndarray,
            next_observation: jnp.ndarray, rng: jnp.ndarray):
    """compute loss for a single transition"""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    # Sample actions with Actor
    _, sampled_action, logp = agent.actor.apply({"params": actor_params}, rng1,
                                                observation)

    # Alpha loss: stop gradient to avoid affect Actor parameters
    log_alpha = agent.log_alpha.apply({"params": alpha_params})
    alpha_loss = -log_alpha * jax.lax.stop_gradient(logp +
                                                    agent.target_entropy)
    alpha = jnp.exp(log_alpha)

    # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
    sampled_q1, sampled_q2 = agent.critic.apply(
        {"params": frozen_critic_params}, observation, sampled_action)
    sampled_q = jnp.squeeze(jnp.minimum(sampled_q1, sampled_q2))

    # Actor loss
    alpha = jax.lax.stop_gradient(
        alpha)  # stop gradient to avoid affect Alpha parameters
    actor_loss = (alpha * logp - sampled_q)

    # Critic loss
    q1, q2 = agent.critic.apply({"params": critic_params}, observation, action)
    q1 = jnp.squeeze(q1)
    q2 = jnp.squeeze(q2)

    # Use frozen_actor_params to avoid affect Actor parameters
    _, next_action, logp_next_action = agent.actor.apply(
        {"params": frozen_actor_params}, rng2, next_observation)
    next_q1, next_q2 = agent.critic.apply({"params": critic_target_params},
                                          next_observation, next_action)

    next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2))
    if agent.backup_entropy:
        next_q -= alpha * logp_next_action
    target_q = reward + agent.gamma * discount * next_q
    critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2

    # CQL loss
    rng3, rng4 = jax.random.split(rng, 2)
    cql_random_actions = jax.random.uniform(rng3,
                                            shape=(agent.num_random,
                                                   agent.act_dim),
                                            minval=-1.0,
                                            maxval=1.0)

    # Sample 10 actions with current state
    repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0),
                                     repeats=agent.num_random,
                                     axis=0)
    repeat_next_observations = jnp.repeat(jnp.expand_dims(next_observation,
                                                          axis=0),
                                          repeats=agent.num_random,
                                          axis=0)
    _, cql_sampled_actions, cql_logp = agent.actor.apply(
        {"params": frozen_actor_params}, rng3, repeat_observations)
    _, cql_next_actions, cql_logp_next_action = agent.actor.apply(
        {"params": frozen_actor_params}, rng4, repeat_next_observations)

    cql_random_q1, cql_random_q2 = agent.critic.apply(
        {"params": critic_params}, repeat_observations, cql_random_actions)
    cql_q1, cql_q2 = agent.critic.apply({"params": critic_params},
                                        repeat_observations,
                                        cql_sampled_actions)
    cql_next_q1, cql_next_q2 = agent.critic.apply({"params": critic_params},
                                                  repeat_observations,
                                                  cql_next_actions)

    random_density = np.log(0.5**agent.act_dim)
    cql_concat_q1 = jnp.concatenate([
        jnp.squeeze(cql_random_q1) - random_density,
        jnp.squeeze(cql_next_q1) - cql_logp_next_action,
        jnp.squeeze(cql_q1) - cql_logp
    ])
    cql_concat_q2 = jnp.concatenate([
        jnp.squeeze(cql_random_q2) - random_density,
        jnp.squeeze(cql_next_q2) - cql_logp_next_action,
        jnp.squeeze(cql_q2) - cql_logp
    ])

    # CQL0: conservative penalty
    logsumexp_cql_concat_q1 = jax.scipy.special.logsumexp(cql_concat_q1)
    logsumexp_cql_concat_q2 = jax.scipy.special.logsumexp(cql_concat_q2)

    # CQL1: maximize Q(s, a) in the dataset
    # cql1_loss = (logsumexp_cql_concat_q1 - q1) * agent.min_q_weight
    # cql2_loss = (logsumexp_cql_concat_q2 - q2) * agent.min_q_weight

    cql1_loss = logsumexp_cql_concat_q1 * agent.min_q_weight
    cql2_loss = logsumexp_cql_concat_q2 * agent.min_q_weight

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
        "logsumexp_cql_concat_q1": logsumexp_cql_concat_q1,
        "logsumexp_cql_concat_q2": logsumexp_cql_concat_q2,
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


grad_fn = jax.vmap(jax.value_and_grad(loss_fn, argnums=(0, 1, 2),
                                      has_aux=True),
                   in_axes=(None, None, None, 0, 0, 0, 0, 0, 0))
rng = jnp.stack(jax.random.split(jax.random.PRNGKey(0), num=actions.shape[0]))
(_, log_info), gradients = grad_fn(agent.actor_state.params,
                                   agent.critic_state.params,
                                   agent.alpha_state.params, observations,
                                   actions, rewards, discounts,
                                   next_observations, rng)
