from typing import Any, Dict, Tuple
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import jax
import jax.numpy as jnp
import ml_collections
import gym
import d4rl
import optax
import time
import logging
import numpy as np
import pandas as pd
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
from tqdm import trange
from models import Actor, DoubleCritic
from utils import ReplayBuffer, Batch


@jax.jit
def sample_action(actor_state: train_state.TrainState, observation: np.ndarray) -> np.ndarray:
    sampled_action = actor_state.apply_fn({"params": actor_state.params}, observation)
    return sampled_action


def eval_policy(actor_state: train_state.TrainState, env: gym.Env, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = sample_action(actor_state, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


# initialize the d4rl environment 
env = gym.make("halfcheetah-medium-v2")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

rng = jax.random.PRNGKey(0)
actor_rng, critic_rng = jax.random.split(rng, 2)

dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

actor = Actor(act_dim=act_dim)
actor_params = actor.init(actor_rng, dummy_obs)["params"]
actor_target_params = actor_params
actor_state = train_state.TrainState.create(
    apply_fn=actor.apply,
    params=actor_params,
    tx=optax.adam(learning_rate=3e-4))

critic = DoubleCritic()
critic_params = critic.init(critic_rng, dummy_obs, dummy_act)["params"]
critic_target_params = critic_params
critic_state = train_state.TrainState.create(
    apply_fn=critic.apply,
    params=critic_params,
    tx=optax.adam(learning_rate=3e-4))

# replay buffer
replay_buffer = ReplayBuffer(obs_dim, act_dim)
logs = [{"step":0, "reward":eval_policy(actor_state, env)[0]}]


obs, done = env.reset(), False
episode_num = 0
episode_reward = 0
episode_timesteps = 0

for _ in range(3000):
    action = (sample_action(actor_state, obs).to_py() +
            np.random.normal(0, max_action*0.2,
                            size=act_dim)).clip(-max_action, max_action)

    next_obs, reward, done, _ = env.step(action)
    done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

    replay_buffer.add(obs, action, next_obs, reward, done_bool)
    obs = next_obs
    episode_reward += reward

    if done:
        obs, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

batch = replay_buffer.sample(256)

# critic_state, log_info = train_step(batch, critic_key, actor_state, critic_state,
#                                     actor_target_params, critic_target_params,
#                                     configs.policy_noise, configs.noise_clip,
#                                     max_action, configs.gamma) 

# actor_state, critic_state, actor_target_params, critic_target_params, log_info = delayed_train_step(
#     batch, critic_key, actor_state, critic_state, actor_target_params, critic_target_params,
#     configs.policy_noise, configs.noise_clip, max_action, configs.gamma, configs.tau)

critic_rng, critic_key = jax.random.split(critic_rng, 2)
noise = jax.random.normal(critic_key, batch.actions.shape) * 0.1
noise = jnp.clip(noise, -0.5, 0.5)
next_actions = actor_state.apply_fn({"params": actor_target_params}, batch.next_observations)
next_actions = jnp.clip(next_actions+noise, -max_action, max_action)

next_q1, next_q2 = critic_state.apply_fn({"params": critic_target_params}, batch.next_observations,
                                         next_actions)
next_q = jnp.minimum(next_q1, next_q2)
target_q = batch.rewards + 0.99 * batch.discounts * next_q

q1, q2 = critic_state.apply_fn({"params": critic_state.params}, batch.observations, batch.actions)
critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

actions = actor_state.apply_fn({"params": actor_params}, batch.observations)
q = critic_state.apply_fn({"params": critic_params}, batch.observations, actions, method=DoubleCritic.Q1)
actor_loss = -q.mean()
