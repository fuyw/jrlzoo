import os
import gym
import sys
import train
from absl import app, flags
from ml_collections import config_flags
from atari_wrappers import wrap_deepmind

config_flags.DEFINE_config_file("config", default="configs/atari.py")
FLAGS = flags.FLAGS
FLAGS(sys.argv)
config = FLAGS.config


env = gym.make(f"{config.env_name}NoFrameskip-v4")
env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")
eval_env = gym.make(f"{config.env_name}NoFrameskip-v4")
eval_env = wrap_deepmind(eval_env, dim=84, obs_format="NHWC", test=True)
act_dim = env.action_space.n
