import jax.numpy as jnp

observations = replay_buffer.sample(agent.rollout_batch_size).observations
sample_rng = jnp.stack(jax.random.split(agent.rollout_rng, num=agent.rollout_batch_size))
select_action = jax.vmap(agent.select_action, in_axes=(None, 0, 0, None))

for t in range(5):
    agent.rollout_rng, rollout_key = jax.random.split(agent.rollout_rng, 2)  # (2,)
    sample_rng, actions = select_action(agent.actor_state.params, sample_rng, observations, False)
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


######
import gym
import d4rl
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from models import GaussianMLP
env = gym.make('hopper-medium-v2')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
ensemble_num = 7

rng = jax.random.PRNGKey(0)
_, model_key = jax.random.split(rng, 2)
model = GaussianMLP(ensemble_num=ensemble_num, out_dim=obs_dim+1)
dummy_model_inputs = jnp.ones([ensemble_num, obs_dim+act_dim], dtype=jnp.float32)
model_params = model.init(model_key, dummy_model_inputs)["params"]
model_state = train_state.TrainState.create(
    apply_fn=model.apply, params=model_params,
    tx=optax.adamw(learning_rate=1e-3))

x = np.random.normal(size=(7, 14))
mu, log_var = model.apply({'params': model_state.params}, x)  # (7, 14) ==> (7, 12), (7, 12)


def val_loss_fn(params, x, y):
    mu, log_var = jax.lax.stop_gradient(model.apply({"params": params}, x))
    inv_var = jnp.exp(-log_var)
    val_loss = jnp.mean(jnp.square(mu - y) * inv_var, axis=-1)
    mse_loss = jnp.mean(jnp.mean(jnp.square(mu - y), axis=-1), axis=-1)
    return val_loss, {"mse_loss": mse_loss}
val_loss_fn = jax.vmap(val_loss_fn, in_axes=(None, 1, 1))

val_X = np.random.normal(size=(7, 128, 14))
val_Y = np.random.normal(size=(7, 128, 12))

A = val_loss_fn(model_state.params, val_X, val_Y)
