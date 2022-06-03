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


def eval_policy(agent: TD3Agent, env: gym.Env, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(agent.actor_state.params, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward, time.time() - t1


def train_and_evaluate(configs: ml_collections.ConfigDict): 
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f'td3_s{configs.seed}_{timestamp}'
    exp_info = f'# Running experiment for: {exp_name}_{configs.env_name} #'
    ckpt_dir = f"{configs.model_dir}/{configs.env_name.lower()}/{exp_name}"
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    logger = get_logger(f'logs/{configs.env_name.lower()}/{exp_name}.log')
    logger.info(f"Exp configurations:\n{configs}")

    # initialize the d4rl environment 
    env = gym.make(configs.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent = TD3Agent(obs_dim=obs_dim,
                     act_dim=act_dim,
                     max_action=max_action,
                     tau=configs.tau,
                     gamma=configs.gamma,
                     noise_clip=configs.noise_clip,
                     policy_noise=configs.policy_noise,
                     policy_freq=configs.policy_freq,
                     lr=configs.lr,
                     seed=configs.seed,
                     hidden_dims=configs.hidden_dims,
                     initializer=configs.initializer)

    # replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    logs = [{"step":0, "reward":eval_policy(agent, env, configs.eval_episodes)[0]}]

    obs, done = env.reset(), False
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    for t in trange(1, configs.max_timesteps+1):
        episode_timesteps += 1
        if t <= configs.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (agent.sample_action(
                agent.actor_state.params, obs) + np.random.normal(
                    0, max_action * configs.expl_noise,
                    size=act_dim)).clip(-max_action, max_action)

        next_obs, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs
        episode_reward += reward

        if t > configs.start_timesteps:
            batch = replay_buffer.sample(configs.batch_size)
            log_info = agent.update(batch)

        if done:
            obs, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if t % configs.eval_freq == 0:
            eval_reward, eval_time = eval_policy(agent, env, configs.eval_episodes)
            if t > configs.start_timesteps:
                log_info.update({
                    "step": t,
                    "reward": eval_reward,
                    "eval_time": eval_time,
                    "time": (time.time() - start_time) / 60
                })
                logger.info(
                    f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}, time: {log_info['time']:.2f}\n"
                    f"\tcritic_loss: {log_info['critic_loss']:.3f}, max_critic_loss: {log_info['max_critic_loss']:.3f}, min_critic_loss: {log_info['min_critic_loss']:.3f}\n"
                    f"\tactor_loss: {log_info['actor_loss']:.3f}, max_actor_loss: {log_info['max_actor_loss']:.3f}, min_actor_loss: {log_info['min_actor_loss']:.3f}\n"
                    f"\tq1: {log_info['q1']:.3f}, max_q1: {log_info['max_q1']:.3f}, min_q1: {log_info['min_q1']:.3f}\n"
                    f"\tq2: {log_info['q2']:.3f}, max_q2: {log_info['max_q2']:.3f}, min_q2: {log_info['min_q2']:.3f}\n"
                    f"\ttarget_q: {log_info['target_q']:.3f}, max_target_q: {log_info['max_target_q']:.3f}, min_target_q: {log_info['min_target_q']:.3f}\n"
                ) 
                logs.append(log_info)
            else:
                logs.append({"step": t, "reward": eval_reward})
                logger.info(
                    f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}\n")

        # Save checkpoints
        if t % configs.ckpt_freq == 0:
            agent.save(f"{ckpt_dir}", t // configs.ckpt_freq)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{configs.log_dir}/{configs.env_name.lower()}/{exp_name}.csv")
