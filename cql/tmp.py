from absl import app, flags
from ml_collections import config_flags
import os
import train
from typing import Any, Dict, Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
import jax
import jax.numpy as jnp
import ml_collections
import gym
import d4rl
import optax
import time
import numpy as np
import pandas as pd
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
from functools import partial
from tqdm import trange
from models import CQLAgent
from utils import ReplayBuffer, Batch, get_logger
import functools



# config_flags.DEFINE_config_file("config", default="configs/antmaze.py")
config_flags.DEFINE_config_file("config", default="configs/antmaze.py")
FLAGS = flags.FLAGS

import sys
FLAGS(sys.argv)
configs = FLAGS.config

env = gym.make(configs.env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
if configs.target_entropy is None:
    target_entropy = -act_dim
else:
    target_entropy = target_entropy

agent = CQLAgent(obs_dim=obs_dim,
                 act_dim=act_dim,
                 max_action=max_action,
                 seed=configs.seed,
                 tau=configs.tau,
                 gamma=configs.gamma,
                 critic_lr=configs.critic_lr,
                 actor_lr=configs.actor_lr,
                 target_entropy=configs.target_entropy,
                 backup_entropy=configs.backup_entropy,
                 max_target_backup=configs.max_target_backup,
                 num_random=configs.num_random,
                 min_q_weight=configs.min_q_weight,
                 bc_timesteps=configs.bc_timesteps,
                 with_lagrange=configs.with_lagrange,
                 lagrange_thresh=configs.lagrange_thresh,
                 cql_clip_diff_min=configs.cql_clip_diff_min,
                 cql_clip_diff_max=configs.cql_clip_diff_max,
                 policy_log_std_multiplier=configs.policy_log_std_multiplier,
                 policy_log_std_offset=configs.policy_log_std_offset,
                 actor_hidden_dims=(256, 256),
                 critic_hidden_dims=(256, 256),
                 initializer=configs.initializer)


agent.load("saved_models/antmaze-large-play-v0/cql_s0_L2/", 50)

# replay buffer
replay_buffer = ReplayBuffer(obs_dim, act_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
batch = replay_buffer.sample(200)
log_info = agent.update(batch)
key = jax.random.PRNGKey(100)
cql_alpha = agent.log_cql_alpha.apply({"params": agent.cql_alpha_state.params})
res = agent.cql_train_step(batch, key, agent.alpha_state, agent.actor_state, agent.critic_state,
                           agent.critic_target_params, cql_alpha, False)
log_info = res[-1]
log_cql_alpha = agent.log_cql_alpha.apply({"params": agent.cql_alpha_state.params})
res_lag = agent.lagrange_train_step(agent.cql_alpha_state, log_info["cql_diff1"], log_info["cql_diff2"])
print(res_lag[-1])