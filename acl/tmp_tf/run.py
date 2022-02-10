"""
python3 -m batch_rl.train_eval_offline \
  --downstream_mode offline \
  --algo_name=brac \
  --task_name ant-medium-v0 \
  --embed_learner acl \
  --state_embed_dim 256 \
  --embed_training_window 8 \
  --embed_pretraining_steps 200_000 \
  --alsologtostderr
"""

import os

import gym
import numpy as np

from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import tqdm
import time

from batch_rl import behavioral_cloning
from batch_rl import latent_behavioral_cloning
from batch_rl import brac
from batch_rl import d4rl_utils
from batch_rl import evaluation
from batch_rl import sac
from batch_rl import embed
from batch_rl import action_embed
