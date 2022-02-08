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


def eval_policy(agent: CDCAgent,
                env_name: str,
                seed: int,
                eval_episodes: int = 10) -> float:
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    eval_rng = jax.random.PRNGKey(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        t = 0
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            eval_rng, action = agent.select_action(agent.actor_state.params,
                                                   eval_rng, np.array(obs), True)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


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

        # 1209 [0.6336192  0.76014924 1.3123713  0.47833192 0.4488593  0.55731165]
        # critic_loss: 3.09 actor_loss: -31.43, mle_prob: 4.13 penalty_loss: 0.41
        # concat_q_avg: 30.96, concat_q_min: 30.74, concat_q_max: 31.14
T = 1

# Train
for _ in range(10):
    for t in range(10):
        log_info = agent.update(replay_buffer, 256)
        mu, pi_distribution = agent.actor.apply({"params": agent.actor_state.params},
                                                next_observation)
        print(f'\t{T+t}', mu)
        print(
            f"\tcritic_loss: {log_info['critic_loss']:.2f} actor_loss: {log_info['actor_loss']:.2f}, mle_prob: {log_info['mle_prob']:.2f} penalty_loss: {log_info['penalty_loss']:.2f}\n"
            f"\tconcat_q_avg: {log_info['concat_q_avg']:.2f}, concat_q_min: {log_info['concat_q_min']:.2f}, concat_q_max: {log_info['concat_q_max']:.2f} "
            f"\tsampled_q: {log_info['sampled_q']:.2f}")
    T += 110


#############
frozen_actor_params = agent.actor_state.params
frozen_critic_params = agent.critic_state.params



rng = jax.random.PRNGKey(0)
rng1, rng2, rng3 = jax.random.split(rng, 3)

mu, pi_distribution = agent.actor.apply({"params": actor_params}, observation)
sampled_action, _ = pi_distribution.sample_and_log_prob(seed=rng1)  # (6,)
mle_prob = pi_distribution.log_prob(action)  # (,)

concat_sampled_q = agent.critic.apply(
    {"params": frozen_critic_params}, observation, sampled_action)
sampled_q = agent.nu*concat_sampled_q.min(-1) + (1.-agent.nu)*concat_sampled_q.max(-1)  # ()

# Actor loss
actor_loss = (-agent.lmbda*mle_prob - sampled_q)

# Critic loss
concat_q = agent.critic.apply({"params": critic_params}, observation, action)  # (4,)

repeat_next_observations = jnp.repeat(jnp.expand_dims(next_observation, axis=0),
                                      repeats=agent.num_samples, axis=0)  # (15, 17)

_, next_pi_distribution = agent.actor.apply({"params": frozen_actor_params},
                                           repeat_next_observations)
sampled_next_actions = next_pi_distribution.sample(seed=rng2)  # (15, 6)

concat_next_q = agent.critic.apply(
    {"params": agent.critic_target_params}, repeat_next_observations, sampled_next_actions)  # (15, 4)

weighted_next_q = agent.nu * concat_next_q.min(-1) + (1. - agent.nu) * concat_next_q.max(-1)  # (15,)

next_q = weighted_next_q.max(-1)  # ()
target_q = reward + agent.gamma * discount * next_q  # ()

critic_loss = jnp.square(concat_q - target_q).sum()

# Overestimation penalty loss
repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0),
                                 repeats=agent.num_samples, axis=0)  # (15, 17)
_, penalty_pi_distribution = agent.actor.apply({"params": agent.actor_state.params},
                                                repeat_observations)
penalty_sampled_actions = penalty_pi_distribution.sample(seed=rng3)  # (15, 6)
penalty_concat_q = agent.critic.apply({"params": agent.critic_state.params},
    repeat_observations, penalty_sampled_actions).max(0)  # (4,)

delta_concat_q = concat_q.reshape(1, -1) - penalty_concat_q.reshape(-1, 1)  # (4, 4)
penalty_loss = jnp.square(jax.nn.relu(delta_concat_q)).mean()

total_loss = critic_loss + actor_loss + penalty_loss * agent.eta