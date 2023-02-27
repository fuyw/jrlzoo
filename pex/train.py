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
from models import AWACAgent, CQLAgent, IQLAgent, SACAgent
from tqdm import trange
from utils import ReplayBuffer, get_logger, load_ckpt, normalize_reward


def normalize_rewards(replay_buffer: ReplayBuffer, env_name: str):
    if 'v2' in env_name:
        # mujoco environments
        normalize_info_df = pd.read_csv('configs/minmax_traj_reward.csv',
                                        index_col=0).set_index('env_name')
        min_traj_reward, max_traj_reward = normalize_info_df.loc[
            env_name, ['min_traj_reward', 'max_traj_reward']]
        replay_buffer.rewards = replay_buffer.rewards / (
            max_traj_reward - min_traj_reward) * 1000
    else:
        # antmaze environments
        replay_buffer.rewards -= 1.0


def initialize_agent(config, obs_dim, act_dim, max_action):
    if config.algo in ["iql", "nobuffer"]:
        agent = IQLAgent(obs_dim=obs_dim,
                         act_dim=act_dim,
                         max_action=max_action,
                         seed=config.seed,
                         expectile=config.expectile,
                         adv_temperature=config.adv_temperature,
                         max_timesteps=config.max_timesteps)
    elif config.algo == "cql":
        agent = CQLAgent(obs_dim=obs_dim,
                         act_dim=act_dim,
                         max_action=max_action,
                         seed=config.seed)
    elif config.algo == "awac":
        agent = AWACAgent(obs_dim=obs_dim,
                          act_dim=act_dim,
                          max_action=max_action,
                          seed=config.seed)
    elif config.algo == "sac":
        agent = SACAgent(obs_dim=obs_dim,
                         act_dim=act_dim,
                         max_action=max_action,
                         seed=config.seed)
    return agent


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
    if "v2" in config.env_name:
        log_dir = f"logs/exp_mujoco_oniql/ms{config.max_timesteps/10000:.0f}_lb{config.lmbda}_ns{config.nstep}/{config.env_name}"
    else:
        log_dir = f"logs/exp_antmaze_oniql/ms{config.max_timesteps/10000:.0f}_lb{config.lmbda}_ns{config.nstep}/{config.env_name}" 
    os.makedirs(log_dir, exist_ok=True)
    exp_name = f"ms{config.max_timesteps/10000:.0f}_lb{config.lmbda}_ns{config.nstep}_s{config.seed}_{timestamp}"
    exp_info = f"# Running experiment for: {exp_name}_{config.env_name} #"
    print("#" * len(exp_info) + f"\n{exp_info}\n" + "#" * len(exp_info))

    logger = get_logger(f"{log_dir}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    # initialize the environment
    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # initialize agent
    agent = initialize_agent(config, obs_dim, act_dim, max_action)

    # load checkpoint
    load_ckpt(agent, config.base_algo, config.env_name, cnt=200)
    logs = [{
        "step": 0,
        "reward": eval_policy(agent, env, config.eval_episodes)[0]
    }]

    # use offline buffer or not
    # if config.offline_buffer:
    #     replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(2e6))
    #     replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    #     if config.base_algo == "iql":
    #         normalize_rewards(replay_buffer, config.env_name)
    # else:
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

        if "antmaze" in config.env_name:
            reward = normalize_reward(config.env_name, reward)

        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs

        if t > config.start_timesteps:
            batch = replay_buffer.sample(config.batch_size)
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
