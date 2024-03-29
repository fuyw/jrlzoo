from typing import Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
import ml_collections
import gym
import d4rl
import time
import numpy as np
import pandas as pd
from tqdm import trange
from models import TD3BCAgent
from utils import ReplayBuffer, get_logger


def normalize_rewards(replay_buffer: ReplayBuffer, env_name: str):
    if 'v2' in env_name:
        # mujoco environments
        normalize_info_df = pd.read_csv('configs/minmax_traj_reward.csv', index_col=0).set_index('env_name')
        min_traj_reward, max_traj_reward = normalize_info_df.loc[env_name, ['min_traj_reward', 'max_traj_reward']]
        replay_buffer.rewards = replay_buffer.rewards / (max_traj_reward - min_traj_reward) * 1000
    if 'v0' in env_name:
        # antmaze environments
        replay_buffer.rewards -= 1.0


def eval_policy(agent: TD3BCAgent, env: gym.Env, mean: np.ndarray = 0.0,
                std: np.ndarray = 1.0, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            obs = (obs - mean) / std
            action = agent.sample_action(agent.actor_state.params, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


def train_and_evaluate(configs: ml_collections.ConfigDict): 
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f'{configs.algo}_s{configs.seed}_{timestamp}'
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

    agent = TD3BCAgent(obs_dim=obs_dim,
                       act_dim=act_dim,
                       max_action=max_action,
                       tau=configs.tau,
                       gamma=configs.gamma,
                       noise_clip=configs.noise_clip,
                       policy_noise=configs.policy_noise,
                       policy_freq=configs.policy_freq,
                       lr=configs.lr,
                       alpha=configs.alpha,
                       seed=configs.seed,
                       hidden_dims=configs.hidden_dims,
                       initializer=configs.initializer)

    # replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    normalize_rewards(replay_buffer, configs.env_name)
    obs_mean, obs_std = replay_buffer.normalize_obs()

    logs = [{"step":0, "reward":eval_policy(agent, env, obs_mean, obs_std, configs.eval_episodes)[0]}]

    for t in trange(1, configs.max_timesteps+1):
        batch = replay_buffer.sample(configs.batch_size)
        log_info = agent.update(batch)

        # Save every 1e5 steps & last 5 checkpoints
        if (t % 100000 == 0) or (t >= int(9.8e5) and t % configs.eval_freq == 0):
            agent.save(f"{ckpt_dir}", t // configs.eval_freq)

        # two-stage eval_freq to save time, only affects the `mujoco` environments
        if (t>int(9.5e5) and (t % configs.eval_freq == 0)) or (t<=int(9.5e5) and t % (2*configs.eval_freq) == 0):
            eval_reward, eval_time = eval_policy(agent, env, obs_mean, obs_std, configs.eval_episodes)
            log_info.update({"step": t, "reward": eval_reward, "eval_time": eval_time, "time": (time.time()-start_time) / 60})
            logs.append(log_info)
            logger.info(
                f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}, time: {log_info['time']:.2f}\n"
                f"\tcritic_loss: {log_info['critic_loss']:.3f}, actor_loss: {log_info['actor_loss']:.3f}, bc_loss: {log_info['bc_loss']:.3f}\n"
                f"\tq1: {log_info['q1']:.3f}, max_q1: {log_info['max_q1']:.3f}, min_q1: {log_info['min_q1']:.3f}\n"
            ) 

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{configs.log_dir}/{configs.env_name}/{exp_name}.csv")
