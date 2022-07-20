from absl import app, flags
from ml_collections import config_flags
import env_utils
import gym
import sys
import jax
import optax
import numpy as np
import jax.numpy as jnp
from models import PPOAgent, ActorCritic

config_flags.DEFINE_config_file("config", default="configs/atari.py")
FLAGS = flags.FLAGS
FLAGS(sys.argv)
config = FLAGS.config
lr = 1e-3
rng = jax.random.PRNGKey(config.seed)
dummy_obs = jnp.ones([1, 84, 84, 4])
learner = ActorCritic(act_dim=6)
learner_params = learner.init(rng, dummy_obs)["params"]

k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

observations = jax.random.normal(k1, shape=(256, 84, 84, 4))
actions = jax.random.randint(k2, minval=0, maxval=6, shape=(256,))
old_log_probs = jax.random.normal(k3, shape=(256,))
targets = jax.random.normal(k4, shape=(256,))


action_distributions, values = learner.apply({"params": learner_params}, observations)
print(f"values.shape = {values.shape}")  # (256,)
value_loss = jnp.square(targets - values).mean()

entropy = action_distributions.entropy()
print(f"entropy.shape = {entropy.shape}")

log_probs = action_distributions.log_prob(actions)
print(f"log_probs.shape = {log_probs.shape}")

ratios = jnp.exp(log_probs - old_log_probs)
print(f"ratios.shape = {ratios.shape}")

pg_loss = ratios * advantages
print(f"pg_loss")

