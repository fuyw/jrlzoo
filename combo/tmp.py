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