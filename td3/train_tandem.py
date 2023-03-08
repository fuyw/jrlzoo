"""Online TD3 Agent (~1000fps)
"""
from typing import Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import ml_collections
import gym
import time
import numpy as np
import pandas as pd
from tqdm import trange
from models import TD3Agent
from utils import ReplayBuffer, get_logger
from gym_utils import make_env


def eval_policy(agent: TD3Agent, env: gym.Env, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward, time.time() - t1


def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f'td3_s{config.seed}_{timestamp}'
    exp_info = f'# Running experiment for: {exp_name}_{config.env_name} #'
    ckpt_dir = f"{config.model_dir}/{config.env_name.lower()}/{exp_name}"
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    logger = get_logger(f'logs/{config.env_name.lower()}/{exp_name}.log')
    logger.info(f"Exp configurations:\n{config}")

    # initialize the gym/d4rl environment
    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)

    # initialize dm_control environment
    # env = make_env(config.env_name, config.seed)
    # eval_env = make_env(config.env_name, config.seed + 42)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent1 = TD3Agent(obs_dim=obs_dim,
                      act_dim=act_dim,
                      max_action=max_action,
                      tau=config.tau,
                      gamma=config.gamma,
                      noise_clip=config.noise_clip,
                      policy_noise=config.policy_noise,
                      policy_freq=config.policy_freq,
                      lr=config.lr,
                      seed=config.seed,
                      hidden_dims=config.hidden_dims,
                      initializer=config.initializer)

    agent2 = TD3Agent(obs_dim=obs_dim,
                      act_dim=act_dim,
                      max_action=max_action,
                      tau=config.tau,
                      gamma=config.gamma,
                      noise_clip=config.noise_clip,
                      policy_noise=config.policy_noise,
                      policy_freq=config.policy_freq,
                      lr=config.lr,
                      seed=config.seed+42,
                      hidden_dims=config.hidden_dims,
                      initializer=config.initializer)

    # replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    logs = [{"step":0, "reward_1":eval_policy(agent1, eval_env, config.eval_episodes)[0],
             "reward_2":eval_policy(agent2, eval_env, config.eval_episodes)[0]}]

    obs = env.reset()
    for t in trange(1, config.max_timesteps+1):
        if t <= config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (agent1.sample_action(obs) + np.random.normal(
                0, max_action * config.expl_noise,
                size=act_dim)).clip(-max_action, max_action)

        next_obs, reward, done, info = env.step(action)
        done_bool = float(done) if "TimeLimit.truncated" not in info else 0
        # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs

        if t > config.start_timesteps:
            batch = replay_buffer.sample(config.batch_size)
            log_info1 = agent1.update(batch)
            log_info2 = agent2.update(batch)
            log_info = {}
            for k,v in log_info1.items():
                log_info[f"{k}_1"] = v
            for k,v in log_info2.items():
                log_info[f"{k}_2"] = v

        if done:
            obs, done = env.reset(), False

        if ((t>int(9.5e5) and (t % config.eval_freq == 0)) or (
                t<=int(9.5e5) and t % (2*config.eval_freq) == 0)):
            eval_reward1, eval_time1 = eval_policy(agent1, eval_env, config.eval_episodes)
            eval_reward2, eval_time2 = eval_policy(agent2, eval_env, config.eval_episodes)
            if t > config.start_timesteps:
                log_info.update({
                    "step": t,
                    "reward_1": eval_reward1,
                    "reward_2": eval_reward2,
                    "eval_time_1": eval_time1,
                    "eval_time_2": eval_time2,
                    "time": (time.time() - start_time) / 60
                })
                logger.info(
                    f"\n[#Step {t}] eval_reward1: {eval_reward1:.2f}, eval_reward2: {eval_reward2:.2f}, time: {log_info['time']:.2f}\n"
                    f"\tcritic_loss_1: {log_info['critic_loss_1']:.3f}, actor_loss_1: {log_info['actor_loss_1']:.3f}\n"
                    f"\tcritic_loss_2: {log_info['critic_loss_2']:.3f}, actor_loss_2: {log_info['actor_loss_2']:.3f}\n"
                    f"\tq1_1: {log_info['q1_1']:.3f}, max_q1_1: {log_info['max_q1_1']:.3f}, min_q1_1: {log_info['min_q1_1']:.3f}\n"
                    f"\tq1_2: {log_info['q1_2']:.3f}, max_q1_2: {log_info['max_q1_2']:.3f}, min_q1_2: {log_info['min_q1_2']:.3f}\n"
                )
                logs.append(log_info)
            else:
                logs.append({"step": t, "reward_1": eval_reward1, "reward_2": eval_reward2})
                logger.info(
                    f"\n[#Step {t}] eval_reward1: {eval_reward1:.2f}, eval_reward2: {eval_reward2:.2f}\n")

        # Save checkpoints
        # if t % config.ckpt_freq == 0:
        #     agent.save(f"{ckpt_dir}", t // config.ckpt_freq)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{config.log_dir}/{config.env_name.lower()}/{exp_name}.csv")