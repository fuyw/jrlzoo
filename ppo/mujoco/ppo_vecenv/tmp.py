import sys

import gym
from absl import flags
from ml_collections import config_flags

import env_utils
from models import PPOAgent
from utils import ExpTuple, PPOBuffer

config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS
FLAGS(sys.argv)
config = FLAGS.config

train_envs = env_utils.create_vec_env("HalfCheetah-v2",
                                      num_envs=config.actor_num)
eval_env = gym.make("HalfCheetah-v2")
act_dim = eval_env.action_space.shape[0]
obs_dim = eval_env.observation_space.shape[0]
agent = PPOAgent(config, obs_dim=obs_dim, act_dim=act_dim, lr=1e-3)
buffer = PPOBuffer(obs_dim, act_dim, config.rollout_len, config.actor_num,
                   config.gamma, config.lmbda)

observations = train_envs.reset()  # (10, 17)
# actions.shape = (5, 6)
# log_probs.shape = (5,)
# values.shape = (5,)

all_experiences = []
for _ in range(126):
    actions, log_probs, values = agent.sample_actions(observations)
    next_observations, rewards, dones, infos = train_envs.step(actions)
    experiences = [
        ExpTuple(observations[i], actions[i], rewards[i], values[i],
                 log_probs[i], dones[i]) for i in range(config.actor_num)
    ]
    all_experiences.append(experiences)
    observations = next_observations
