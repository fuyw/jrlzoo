import gym
import d4rl
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from models import GaussianMLP, COMBOAgent
from utils import ReplayBuffer


###################
# GYM Environment #
###################
env = gym.make('hopper-medium-v2')
ensemble_num = 7
act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]


########################
# Initialize the model #
########################
model_key = jax.random.PRNGKey(1)
model = GaussianMLP(ensemble_num=ensemble_num, out_dim=obs_dim+1)
dummy_model_inputs = jnp.ones([ensemble_num, obs_dim+act_dim], dtype=jnp.float32)
model_params = model.init(model_key, dummy_model_inputs)["params"]
model_state = train_state.TrainState.create(
    apply_fn=model.apply, params=model_params,
    tx=optax.adamw(learning_rate=1e-3))


##############
# COMBOAgent #
##############
agent = COMBOAgent(env='hopper-medium-v2', obs_dim=obs_dim, act_dim=act_dim,
                   seed=42, lr=1e-3, lr_actor=1e-3)
agent.model.load(f'ensemble_models/hopper-medium-v2/s2')


###########
# Rollout #
###########
rollout_batch_size = 40
rollout_rng = jax.random.PRNGKey(2)
replay_buffer = ReplayBuffer(obs_dim, act_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
model_buffer = ReplayBuffer(obs_dim, act_dim)


observations = replay_buffer.sample(rollout_batch_size).observations  # (100, 11)
sample_rng = jnp.stack(jax.random.split(rollout_rng, num=rollout_batch_size))  # (100, 2)
select_action = jax.vmap(agent.select_action, in_axes=(None, 0, 0, None))

for t in range(5):
    agent.rollout_rng, rollout_key = jax.random.split(agent.rollout_rng, 2)  # (2,)
    sample_rng, actions = select_action(agent.actor_state.params, sample_rng, observations,
                                        False)  # (100, 2),  (100, 3)
    next_observations, rewards, dones = agent.model.step(rollout_key, observations, actions)
    nonterminal_mask = ~dones
    if nonterminal_mask.sum() == 0:
        print(f'[ Model Rollout ] Breaking early {nonterminal_mask.shape}')
        break
    model_buffer.add_batch(observations[nonterminal_mask],
                           actions[nonterminal_mask],
                           next_observations[nonterminal_mask],
                           rewards[nonterminal_mask],
                           dones[nonterminal_mask])
    observations = next_observations[nonterminal_mask]
    sample_rng = sample_rng[nonterminal_mask]

size = model_buffer.size
print('model_buffer.size =', size)
print('model_buffer.observations.sum() =', model_buffer.observations.reshape(-1).sum())
print('model_buffer.actions.sum() =', model_buffer.actions.reshape(-1).sum())
print('model_buffer.rewards =', model_buffer.rewards.reshape(-1).sum())

"""
model_buffer.size = 191
model_buffer.observations.sum() = -180.0183590060151
model_buffer.actions.sum() = -50.49959243182093
model_buffer.rewards = 559.0667119026184
"""


def check_agent():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from models import COMBOAgent
    obs_dim, act_dim = 11, 3
    agent = COMBOAgent(env='hopper-medium-v2', obs_dim=obs_dim, act_dim=act_dim, seed=42, lr=1e-3, lr_actor=1e-3)
    agent.model.load(f'ensemble_models/hopper-medium-v2/s2')
    frozen_actor_params = agent.actor_state.params
    frozen_critic_params = agent.critic_state.params
    actor_params = agent.actor_state.params
    critic_params = agent.critic_state.params
    alpha_params = agent.alpha_state.params
    critic_target_params = agent.critic_state.params

    observations = np.random.normal(size=(32, 11))
    actions = np.random.normal(size=(32, 3))
    rewards = np.random.normal(size=(32,))
    next_observations = np.random.normal(size=(32, 11))
    observation = observations[0]
    action = actions[0]
    next_observation = next_observations[0]
    reward = rewards[0]

    rng = jax.random.PRNGKey(0)
    rng, rng1, rng2 = jax.random.split(rng, 3)

    # (11) ==> (3), ()
    _, sampled_action, logp = agent.actor.apply({"params": actor_params}, rng1, observation)

    # Alpha loss ==> ()
    log_alpha = agent.log_alpha.apply({"params": alpha_params})
    alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + agent.target_entropy)
    alpha = jnp.exp(log_alpha)

    # Sampled Q ==> (1)
    sampled_q1, sampled_q2 = agent.critic.apply({"params": frozen_critic_params},
                                                observation, sampled_action)
    sampled_q = jnp.squeeze(jnp.minimum(sampled_q1, sampled_q2))  # ()

    # Actor loss ==> ()
    alpha = jax.lax.stop_gradient(alpha)
    actor_loss = (alpha * logp - sampled_q)

    # Critic loss
    q1, q2 = agent.critic.apply({"params": critic_params}, observation, action)  # (1), (1)
    q1, q2 = jnp.squeeze(q1), jnp.squeeze(q2)                             # ()
    _, next_action, logp_next_action = agent.actor.apply(
        {"params": frozen_actor_params}, rng2, next_observation)          # (3), ()
    next_q1, next_q2 = agent.critic.apply(
        {"params": critic_target_params}, next_observation, next_action)  # (1), (1)
    next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2))                   # ()
    target_q = reward + 0.99 * next_q                                     # ()
    critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2                 # ()

    # COMBO CQL loss
    rng3, rng4 = jax.random.split(rng, 2)
    cql_random_actions = jax.random.uniform(rng3, shape=(10, 3), minval=-1.0, maxval=1.0)  # (10, 3)
    repeat_next_observations = jnp.repeat(jnp.expand_dims(next_observation, axis=0),
                                          repeats=10, axis=0)  # (10, 11)
    cql_random_q1, cql_random_q2 = agent.critic.apply({"params": critic_params},
                                                       repeat_next_observations,
                                                       cql_random_actions)  # (10, 1),  (10, 1)
    _, cql_next_actions, cql_logp_next_action = agent.actor.apply(
        {"params": frozen_actor_params}, rng4, repeat_next_observations)  # (10, 3), (10)

    cql_next_q1, cql_next_q2 = agent.critic.apply(
        {"params": critic_params}, repeat_next_observations, cql_next_actions)  # (10, 1)
    
    random_density = np.log(0.5 ** 3)
    cql_concat_q1 = jnp.concatenate([
        jnp.squeeze(cql_random_q1) - random_density,
        jnp.squeeze(cql_next_q1) - cql_logp_next_action,
    ])  # (20,)
    cql_concat_q2 = jnp.concatenate([
        jnp.squeeze(cql_random_q2) - random_density,
        jnp.squeeze(cql_next_q2) - cql_logp_next_action,
    ])  # (20,)

    cql1_loss = jax.scipy.special.logsumexp(cql_concat_q1) - q1
