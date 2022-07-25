import jax
import jax.numpy as jnp
import distrax
import optax
from flax.training import train_state
from models import ActorCritic
import numpy as np
import envpool



rng = jax.random.PRNGKey(0)
model = ActorCritic(act_dim=6)
dummy_obs = jnp.ones([1, 17])
params = model.init(rng, rng, dummy_obs)["params"]
learner_state = train_state.TrainState.create(apply_fn=ActorCritic.apply, params=params, tx=optax.adam(1e-3))

env = envpool.make_gym("HalfCheetah-v3", num_envs=5)   
observations = env.reset()  # (5, 17)
actions = np.concatenate([env.action_space.sample().reshape(1, -1) for _ in range(5)], 0)
next_observations, rewards, dones, infos = env.step(actions)


# mean_actions, sampled_actions, log_probs, values = model.apply(
#             {"params": params}, rng, obs)

# # mean_actions.shape = (5, 6)
# # sampled_actions.shape = (5, 6)
# # logps.shape = (5,)
# # values.shape = (5,)

# def atanh(x: jnp.ndarray):
#     one_plus_x = jnp.clip(1 + x, a_min=1e-6)
#     one_minus_x = jnp.clip(1 - x, a_min=1e-6)
#     return 0.5 * jnp.log(one_plus_x / one_minus_x)


# rng, k1, k2, k3 =jax.random.split(rng, 4)
# mu = jax.random.normal(k1, shape=(5, 6))
# log_std = jax.random.normal(k2, shape=(5, 6))
# actions = jax.random.normal(k3, shape=(5, 6))
# std = jnp.exp(log_std)

# pi = distrax.Transformed(
#     distrax.MultivariateNormalDiag(mu, std),
#     distrax.Block(distrax.Tanh(), ndims=1))
# a, logp = pi.sample_and_log_prob(seed=rng)

# # raw_actions = atanh(actions)  # (5, 6)
# # log_probs1 = pi1.log_prob(raw_actions).sum(axis=-1)
# # log_probs1 -= 2*(jnp.log(2) - raw_actions - jax.nn.softplus(-2*raw_actions))
