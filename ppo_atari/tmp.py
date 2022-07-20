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
# agent = PPOAgent(config, 6, 1e-3)
lr = 1e-3
rng = jax.random.PRNGKey(config.seed)
dummy_obs = jnp.ones([1, 84, 84, 4])
learner = ActorCritic(act_dim=6)
learner_params = learner.init(rng, dummy_obs)["params"]

learner.apply({"params": params}, observation)
