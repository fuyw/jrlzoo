observations = replay_pool.random_batch(agent.rollout_batch_size)['observations']  # (N, 11)
for t in range(agent.horizon):
    actions = np.random.uniform(-1, 1, size=(len(observations), 3))
    next_observations, rewards, dones, info = agent.model.step(observations, actions, False)
    samples = {'observations': observations, 'actions': actions, 'next_observations': next_observations, 'rewards': rewards, 'terminals': dones}
    nonterminal_mask = (~dones).squeeze()
    if nonterminal_mask.sum() == 0: break
    model_pool.add_samples(samples)
    observations = next_observations[nonterminal_mask]
    print(f"model_pool._size = {model_pool._size}")
    print(f"model_pool._ptr = {model_pool._pointer}")


# sample from real & model buffer
real_batch = replay_pool.random_batch(agent.real_batch_size)
model_batch = model_pool.random_batch(agent.model_batch_size)

concat_observations = np.concatenate([
    real_batch['observations'], model_batch['observations']], axis=0)
concat_actions = np.concatenate([
    real_batch['actions'], model_batch['actions']], axis=0)
concat_rewards = np.concatenate([
    real_batch['rewards'].squeeze(), model_batch['rewards'].squeeze()], axis=0)
concat_discounts = np.concatenate([
    1 - real_batch['terminals'].squeeze(), 1 - model_batch['terminals'].squeeze()], axis=0)
concat_next_observations = np.concatenate([
    real_batch['next_observations'], model_batch['next_observations']], axis=0)

rng, key = jax.random.split(agent.rng)


observations = concat_observations  # (256, 11)
actions = concat_actions  # (256, 3)
rewards = concat_rewards  # (256,)
discounts = concat_discounts  # (256,)
next_observations = concat_next_observations  # (256, 11)
critic_target_params = agent.critic_target_params
actor_state = agent.actor_state
critic_state = agent.critic_state
alpha_state = agent.alpha_state

frozen_actor_params = actor_state.params
frozen_critic_params = critic_state.params


actor_params = actor_state.params
critic_params = critic_state.params
alpha_params = alpha_state.params

rng, rng1, rng2 = jax.random.split(rng, 3)
observation = observations[0]
action = actions[0]
next_observation = next_observations[0]
discount = discounts[0]
reward = rewards[0]
mask = agent.masks[0]

# (3,), ()
_, sampled_action, logp = agent.actor.apply({"params": actor_params}, rng1, observation)

log_alpha = agent.log_alpha.apply({"params": alpha_params})  # ()
alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + agent.target_entropy)  # ()
alpha = jnp.exp(log_alpha)
alpha = jax.lax.stop_gradient(alpha)

sampled_q1, sampled_q2 = agent.critic.apply({"params": frozen_critic_params},
                                            observation, sampled_action)  # [1], [1]
sampled_q = jnp.squeeze(jnp.minimum(sampled_q1, sampled_q2))  # ()

actor_loss = (alpha * logp - sampled_q)  # ()

q1, q2 = agent.critic.apply({"params": critic_params}, observation, action)
q1, q2 = jnp.squeeze(q1), jnp.squeeze(q2)

_, next_action, logp_next_action = agent.actor.apply({"params": frozen_actor_params}, rng2, next_observation)

next_q1, next_q2 = agent.critic.apply({"params": critic_target_params},
    next_observation, next_action)
next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2))
target_q = reward + agent.gamma * discount * next_q

critic_loss1 = 0.5 * (q1 - target_q)**2
critic_loss2 = 0.5 * (q2 - target_q)**2

critic_loss = critic_loss1 + critic_loss2

# CQL loss
rng3, rng4 = jax.random.split(rng, 2)
cql_random_actions = jax.random.uniform(
    rng3, shape=(agent.num_random, agent.act_dim), minval=-1.0, maxval=1.0)
repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0),
    repeats=agent.num_random, axis=0)

_, cql_sampled_actions, cql_logp = agent.actor.apply(
    {"params": frozen_actor_params}, rng3, repeat_observations)

cql_random_q1, cql_random_q2 = agent.critic.apply({"params": critic_params},
                                                  repeat_observations,
                                                  cql_random_actions)
cql_q1, cql_q2 = agent.critic.apply({"params": critic_params},
    repeat_observations, cql_sampled_actions)
random_density = np.log(0.5 ** agent.act_dim)
cql_concat_q1 = jnp.concatenate([
    jnp.squeeze(cql_random_q1) - random_density,
    jnp.squeeze(cql_q1) - cql_logp,
])
cql_concat_q2 = jnp.concatenate([
    jnp.squeeze(cql_random_q2) - random_density,
    jnp.squeeze(cql_q2) - cql_logp,
])

ood_q1 = jax.scipy.special.logsumexp(cql_concat_q1)
ood_q2 = jax.scipy.special.logsumexp(cql_concat_q2)

cql1_loss = (ood_q1*(1-mask) - q1*mask) * agent.min_q_weight / agent.real_ratio
cql2_loss = (ood_q2*(1-mask) - q2*mask) * agent.min_q_weight / agent.real_ratio

total_loss = alpha_loss + actor_loss + critic_loss + cql1_loss + cql2_loss


####################
# Run check update #
####################
real_batch = Batch(
    observations=check_data['obs'][agent.update_step][:128],
    actions=check_data['act'][agent.update_step][:128],
    rewards=check_data['rew'][agent.update_step][:128],
    discounts=1 - check_data['done'][agent.update_step][:128],
    next_observations=check_data['next_obs'][agent.update_step][:128]
)
model_batch = Batch(
    observations=check_data['obs'][agent.update_step][128:],
    actions=check_data['act'][agent.update_step][128:],
    rewards=check_data['rew'][agent.update_step][128:],
    discounts=1 - check_data['done'][agent.update_step][128:],
    next_observations=check_data['next_obs'][agent.update_step][128:]
)



