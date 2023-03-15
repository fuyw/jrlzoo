"""Run PEX"""

import os
from typing import Tuple

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import time

import d4rl
import gym
import ml_collections
import numpy as np
import pandas as pd
from models import PEXAgent
from tqdm import trange
from utils import Batch, ReplayBuffer, get_logger, load_ckpt, normalize_reward

normalize_info_df = pd.read_csv('configs/minmax_traj_reward.csv', index_col=0).set_index('env_name')


def normalize_rewards(replay_buffer: ReplayBuffer, env_name: str):
    if 'v2' in env_name:
        # mujoco environments        
        min_traj_reward, max_traj_reward = normalize_info_df.loc[
            env_name, ['min_traj_reward', 'max_traj_reward']]
        replay_buffer.rewards = replay_buffer.rewards / (
            max_traj_reward - min_traj_reward) * 1000
    else:
        # antmaze environments
        replay_buffer.rewards -= 1.0


def eval_policy(agent, env, eval_episodes=10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(obs, eval_mode=True)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"pex_s{config.seed}_{timestamp}"
    log_dir = f"logs/{config.env_name}"
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(f"{log_dir}/{exp_name}.log")

    exp_info = f"# Running experiment for: {exp_name}_{config.env_name} #"
    print("#" * len(exp_info) + f"\n{exp_info}\n" + "#" * len(exp_info))

    logger.info(f"Exp configurations:\n{config}")

    # initialize the environment
    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # initialize agent
    agent = PEXAgent(obs_dim, act_dim, max_action)

    # load checkpoint
    load_ckpt(agent, config.base_algo, config.env_name, cnt=200)
    logs = [{
        "step": 0,
        "reward": eval_policy(agent, env, config.eval_episodes)[0]
    }]

    # use offline buffer or not
    offline_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(1e6))
    offline_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    normalize_rewards(offline_buffer, config.env_name)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(1e6))

    # fine-tuning
    obs, done = env.reset(), False
    episode_timesteps = 0
    for t in trange(1, config.max_timesteps + 1):
        episode_timesteps += 1
        action = agent.sample_action(obs, eval_mode=False)
        next_obs, reward, done, info = env.step(action)
        timeout = "TimeLimit.truncated" in info
        done_bool = float(done) if not timeout else 0.0

        reward = normalize_reward(config.env_name, reward)

        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs

        if t > config.start_timesteps:
            online_batch = replay_buffer.sample(int(config.batch_size*0.75))
            offline_batch = offline_buffer.sample(int(config.batch_size*0.25))
            batch = Batch(observations=np.concatenate([online_batch.observations, offline_batch.observations], 0),
                          actions=np.concatenate([online_batch.actions, offline_batch.actions], 0),
                          rewards=np.concatenate([online_batch.rewards, offline_batch.rewards], 0),
                          discounts=np.concatenate([online_batch.discounts, offline_batch.discounts], 0),
                          next_observations=np.concatenate([online_batch.next_observations, offline_batch.next_observations], 0),
                          idx=np.concatenate([online_batch.idx, offline_batch.idx], 0),
                          flags=np.concatenate([online_batch.flags, offline_batch.flags], 0))
            log_info = agent.update(batch)
            log_info["online_ratio"] = batch.flags.sum() / len(batch.flags)
            sample_age = (replay_buffer.ptr - batch.idx).mean()

        if done:
            obs, done = env.reset(), False
            episode_timesteps = 0

        if t % config.eval_freq == 0:
            eval_reward, eval_time = eval_policy(agent, eval_env,
                                                 config.eval_episodes)
            if t > config.start_timesteps:
                log_info.update({
                    "step": t,
                    "reward": eval_reward,
                    "eval_time": eval_time,
                    "time": (time.time() - start_time) / 60,
                    "sample_age": sample_age,
                    "reset_num": 0,
                    "buffer_size": replay_buffer.size,
                    "buffer_ptr": replay_buffer.ptr,
                })
                logs.append(log_info)
                agent.logger(t, logger, log_info)
            else:
                logs.append({"step": t, "reward": eval_reward})
                logger.info(
                    f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}\n"
                )

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{log_dir}/{exp_name}.csv")
