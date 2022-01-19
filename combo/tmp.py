import d4rl
import gym
from flax import linen as nn
from flax import serialization
from flax.core import FrozenDict
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import os
from utils import ReplayBuffer
from models import COMBOAgent
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr_actor", default=3e-4, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--eval_freq", default=int(5e3), type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--target_entropy", default=None, type=float)
    parser.add_argument("--auto_entropy_tuning", default=True, action="store_false")
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--model_dir", default="./saved_models", type=str)
    parser.add_argument("--backup_entropy", default=False, action="store_true")
    parser.add_argument("--with_lagrange", default=False, action="store_true")
    parser.add_argument("--lagrange_thresh", default=5.0, type=float)
    args = parser.parse_args()
    return args


args = get_args()
env = gym.make(args.env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

replay_buffer = ReplayBuffer(obs_dim, act_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
model_buffer = ReplayBuffer(obs_dim, act_dim)

ensemble_num = 7
rollout_batch_size = 1000
agent = COMBOAgent(args.env, obs_dim, act_dim)
agent.model.load('ensemble_models/hopper-medium-v2/s0')

log_info = agent.update(replay_buffer, model_buffer)

# observations = replay_buffer.sample(agent.rollout_batch_size).observations                 # (N, 11)
# sample_rng = jnp.stack(jax.random.split(agent.rollout_rng, num=agent.rollout_batch_size))  # (N, 2)
# select_action = jax.vmap(agent.select_action, in_axes=(None, 0, 0, None))
# agent.rollout_rng, rollout_key = jax.random.split(agent.rollout_rng, 2)  # (2,)
# sample_rng, actions = select_action(agent.actor_state.params, sample_rng, observations, False)
# next_observations, rewards, dones = agent.model.step(rollout_key, observations, actions)
