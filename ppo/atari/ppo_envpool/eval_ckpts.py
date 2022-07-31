from typing import Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import sys
import time
import envpool
import numpy as np
from models import PPOAgent

from absl import app, flags
from ml_collections import config_flags
import os
import train


def eval_policy(agent: PPOAgent,
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
    eval_step = np.sum(episode_lengths)
    return avg_reward, eval_step, time.time() - t1


config_flags.DEFINE_config_file("config", default="configs/atari.py")
FLAGS = flags.FLAGS
FLAGS(sys.argv)
config = FLAGS.config
env_name = "Pong"
eval_envs = envpool.make(
    task_id=f"{env_name}-v5",
    env_type="gym",
    num_envs=10,
    episodic_life=False,
    reward_clip=False,
)

agent = PPOAgent(config, 6, 3e-4)
for i in range(1, 11):
    agent.load(f"saved_models/{env_name}/ppo_s0", i)
    eval_reward, eval_step, eval_time = eval_policy(agent, eval_envs)
    print(f"Ckpt {i}: {eval_reward:.2f}, {eval_step:.0f}")
