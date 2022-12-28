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
from models import AWACAgent
from utils import ReplayBuffer, get_logger


def eval_policy(agent: AWACAgent,
                env: gym.Env,
                eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(obs, 0.0)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


def train_and_evaluate(configs: ml_collections.ConfigDict): 
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f'awac_s{configs.seed}_{timestamp}'
    exp_info = f'# Running experiment for: {exp_name}_{configs.env_name} #'
    ckpt_dir = f"{configs.model_dir}/{configs.env_name.lower()}/s{configs.seed}"
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    logger = get_logger(f'logs/{configs.env_name.lower()}/{exp_name}.log')
    logger.info(f"Exp configurations:\n{configs}")

    # initialize the d4rl environment 
    env = gym.make(configs.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent = AWACAgent(obs_dim=obs_dim, act_dim=act_dim)

    # replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    logs = [{"step":0, "reward":eval_policy(agent, env, configs.eval_episodes)[0]}]

    for t in trange(1, configs.max_timesteps+1):
        batch = replay_buffer.sample(configs.batch_size)
        log_info = agent.update(batch)

        # two-stage eval_freq to save time, only affects the `mujoco` environments
        if (t>int(9.5e5) and (t % configs.eval_freq == 0)) or (t<=int(9.5e5) and t % (2*configs.eval_freq) == 0):
            eval_reward, eval_time = eval_policy(agent, env, configs.eval_episodes)
            log_info.update({"step": t, "reward": eval_reward, "eval_time": eval_time, "time": (time.time()-start_time) / 60})
            logs.append(log_info)
            logger.info(
                f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}, time: {log_info['time']:.2f}\n"
                f"\tactor_loss: {log_info['actor_loss']:.3f}, actor_loss_max: {log_info['actor_loss_max']:.3f}, actor_loss_min: {log_info['actor_loss_min']:.3f}\n"
                f"\tcritic_loss: {log_info['critic_loss']:.3f}, critic_loss_max: {log_info['critic_loss_max']:.3f}, critic_loss_min: {log_info['critic_loss_min']:.3f}\n"
                f"\tq1: {log_info['q1']:.3f}, q1_max: {log_info['q1_max']:.3f}, q1_min: {log_info['q1_min']:.3f} \n"
                f"\tlogp: {log_info['logp']:.3f}, logp_max: {log_info['logp_max']:.3f}, logp_min: {log_info['logp_min']:.3f}\n"
            )

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{configs.log_dir}/{configs.env_name}/{exp_name}.csv")
