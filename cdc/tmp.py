import gym
import d4rl
import jax
import jax.numpy as jnp
import os
import time
import yaml
import logging
import numpy as np
import pandas as pd
from tqdm import trange
from models import CDCAgent
from utils import ReplayBuffer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=7e-4, type=float)
    parser.add_argument("--lr_actor", default=3e-4, type=float)
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--eval_freq", default=int(5e3), type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--nu", default=0.75, type=float)
    parser.add_argument("--lmbda", default=1.0, type=float)
    parser.add_argument("--eta", default=1.0, type=float)
    parser.add_argument("--num_samples", default=15, type=int)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--config_dir", default="./configs", type=str)
    parser.add_argument("--model_dir", default="./saved_models", type=str)
    args = parser.parse_args()
    return args


args = get_args()
print(f"\nArguments:\n{vars(args)}")
with open(f"{args.config_dir}/cdc.yaml", "r") as stream:
    configs = yaml.safe_load(stream)
args.eta = configs[args.env]["eta"]
args.lmbda = configs[args.env]["lmbda"]


# Env parameters
env = gym.make(args.env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# random seeds
env.seed(args.seed)
env.action_space.seed(args.seed)
np.random.seed(args.seed)


# CDC agent
agent = CDCAgent(obs_dim=obs_dim,
                 act_dim=act_dim,
                 seed=args.seed,
                 nu=args.nu,
                 eta=args.eta,
                 tau=args.tau,
                 gamma=args.gamma,
                 lmbda=args.lmbda,
                 num_samples=args.num_samples,
                 lr=args.lr,
                 lr_actor=args.lr_actor)

# Replay buffer
replay_buffer = ReplayBuffer(obs_dim, act_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
fix_obs = np.random.normal(size=(128, obs_dim))
fix_act = np.random.normal(size=(128, act_dim))


batch = replay_buffer.sample(30)
observations, actions, rewards, discounts, next_observations = batch
actor_params = agent.actor_state.params
critic_params = agent.critic_state.params
observation = observations[0]
action = actions[0]
reward = rewards[0]
discount = discounts[0]
next_observation = next_observations[0]


mean_action, sampled_action = agent.actor.apply(
    {"params": agent.actor_state.params}, observations, jax.random.PRNGKey(0))


repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0),
                                 repeats=15, axis=0)
rng = jax.random.PRNGKey(0)
mean_actions, sampled_actions = agent.actor.apply({"params": actor_params},
                                                  repeat_observations, rng)
concat_qs = agent.critic.apply({"params": critic_params},
                               repeat_observations, sampled_actions)
weighted_q = concat_qs.min(-1) + concat_qs.max(-1)
max_idx = weighted_q.argmax()