from typing import Tuple
import jax
import numpy as np
import env_utils
import envpool
import sys
import time
from absl import flags
from models import PPOAgent
from ml_collections import config_flags
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

def eval_policy1(agent: PPOAgent,
                eval_envs: envpool.atari.AtariGymEnvPool,
                eval_episodes: int = 10) -> Tuple[float]:
    """Evaluate with envpool vectorized environments."""
    t1 = time.time()
    n_envs = len(eval_envs.all_env_ids)

    # record episode reward and length
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    episode_rewards = []
    episode_lengths = []
    episode_counts = np.zeros(n_envs, dtype="int")

    # evaluate `target` episodes for each environment
    episode_count_targets = np.array([(eval_episodes + i) // n_envs
                                      for i in range(n_envs)],
                                     dtype="int")

    # start evaluation
    observations = eval_envs.reset()
    while (episode_counts < episode_count_targets).any():
        # (10, 4, 84, 84) ==> (10, 84, 84, 4)
        actions = agent.sample_actions(np.moveaxis(observations, 1, -1))
        observations, rewards, dones, _ = eval_envs.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
    avg_reward = np.mean(episode_rewards)
    avg_step = np.mean(episode_lengths)
    return avg_reward, avg_step, time.time() - t1


def eval_policy2(agent, env, eval_episodes: int = 10) -> Tuple[float]:
    """For-loop sequential evaluation."""
    t1 = time.time()
    avg_reward = 0.
    avg_step = 0
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            log_probs, _ = agent._sample_actions(agent.learner_state.params,
                                                 obs[None, ...])
            log_probs = jax.device_get(log_probs)
            probs = np.exp(log_probs)  # (1, act_dim)
            action = np.random.choice(probs.shape[1], p=probs[0])
            next_obs, reward, done, _ = env.step(action)
            avg_reward += reward
            avg_step += 1
            obs = next_obs
    avg_reward /= eval_episodes
    avg_step /= eval_episodes
    return avg_reward, avg_step, time.time() - t1


config_flags.DEFINE_config_file("config", default="configs/atari.py")
FLAGS = flags.FLAGS
FLAGS(sys.argv)
config = FLAGS.config


env = envpool.make_gym("Pong-v5", num_envs=10, batch_size=5, episodic_life=False, reward_clip=False)
env.async_reset()
obs, rew, done, info = env.recv()