import sys

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from flax.training import train_state
from ml_collections import config_flags

import env_utils
from models import PPOAgent


config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS
FLAGS(sys.argv)
config = FLAGS.config

rng = jax.random.PRNGKey(0)


env = env_utils.create_vec_env("HalfCheetah-v2", num_envs=5)
act_dim = env.action_space[0].shape[0]
obs_dim = env.observation_space.shape[1]
agent = PPOAgent(config, obs_dim=obs_dim, act_dim=act_dim, lr=1e-3)

observations = env.reset()  # (5, 17)
actions, log_probs, values = agent.sample_actions(observations)

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
