from typing import Dict, Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
import ml_collections
import gym
import d4rl
import time
import numpy as np
import pandas as pd
from tqdm import trange
from models import CDAAgent
from utils import ReplayBuffer, get_logger


def normalize_rewards(replay_buffer: ReplayBuffer, env_name: str):
    if 'v2' in env_name:
        # mujoco environments
        normalize_info_df = pd.read_csv('configs/minmax_traj_reward.csv', index_col=0).set_index('env_name')
        min_traj_reward, max_traj_reward = normalize_info_df.loc[env_name, ['min_traj_reward', 'max_traj_reward']]
        replay_buffer.rewards = replay_buffer.rewards / (max_traj_reward - min_traj_reward) * 1000
        replay_buffer.min_traj_reward = min_traj_reward
        replay_buffer.max_traj_reward = max_traj_reward
    else:
        # antmaze environments
        replay_buffer.rewards -= 1.0


def eval_policy(agent: CDAAgent, env: gym.Env, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(agent.agent.actor_state.params, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


def train_and_evaluate(configs: ml_collections.ConfigDict):
    start_time = time.time()
    exp_name = f'iql_s{configs.seed}_thresh{-configs.var_thresh}'
    exp_info = f'# Running experiment for: {exp_name}_{configs.env_name} #'
    ckpt_dir = f"{configs.model_dir}/{configs.env_name}/s{configs.seed}"
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    logger = get_logger(f'logs/{configs.env_name}/{exp_name}.log')
    logger.info(f"Exp configurations:\n{configs}")

    # initialize the d4rl environment 
    env = gym.make(configs.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = CDAAgent(env_name=configs.env_name,
                     algo=configs.algo,
                     obs_dim=obs_dim,
                     act_dim=act_dim,
                     expectile=configs.expectile,
                     temperature=configs.temperature,
                     var_thresh=configs.var_thresh,
                     initializer=configs.initializer)

    # replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    normalize_rewards(replay_buffer, configs.env_name)
    model_buffer = ReplayBuffer(obs_dim, act_dim, max_size=500000)

    logs = []
    eval_reward, eval_time = 0.0, 0.0
    for t in trange(1, configs.max_timesteps+1):
        log_info = agent.update_agent(replay_buffer, model_buffer, configs.batch_size//2)

        # save checkpoints
        # agent.save(ckpt_dir, t//configs.eval_freq)

        if (t>=int(9.5e5) and (t % configs.eval_freq == 0)) or (t<int(9.5e5) and t % (2*configs.eval_freq) == 0):
            eval_reward, eval_time = eval_policy(agent, env, configs.eval_episodes)
            log_info.update({
                "step": t,
                "reward": eval_reward,
                "eval_time": eval_time,
                "time": (time.time() - start_time) / 60
            })

        # two-stage eval_freq to save time, only affects the `mujoco` environments
        if (t % configs.log_freq == 0):
            logger.info(
                f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}, time: {(time.time()-start_time)/60:.2f}\n"
                f"\tcritic_loss: {log_info['critic_loss']:.2f}, value_loss: {log_info['value_loss']:.2f}, actor_loss: {log_info['actor_loss']:.2f}\n"
                f"\treal_critic_loss: {log_info['real_critic_loss']:.2f}, model_critic_loss: {log_info['model_critic_loss']:.2f}\n"
                f"\tv: {log_info['v']:.2f}, weight: {log_info['weight']:.2f}, adv: {log_info['adv']:.2f}, logp: {log_info['logp']:.2f}\n"
                f"\tq1: {log_info['q1']:.2f}, q2: {log_info['q2']:.2f}, target_q: {log_info['target_q']:.2f}\n"
                f"\treal_batch_obs: {log_info['real_batch_obs']:.2f}, real_batch_act: {log_info['real_batch_act']:.2f}\n"
                f"\tfake_batch_obs: {log_info['model_batch_obs']:.2f}, fake_batch_act: {log_info['model_batch_act']:.2f}\n"
                f"\treal_batch_reward: {log_info['real_batch_reward']:.2f}, real_batch_discount: {log_info['real_batch_discount']:.0f}\n"
                f"\tfake_batch_reward: {log_info['model_batch_reward']:.2f}, fake_batch_discount: {log_info['model_batch_discount']:.0f}\n"
            ) 
            logs.append(log_info)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{configs.log_dir}/{configs.env_name}/{exp_name}.csv")
