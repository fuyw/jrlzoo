from typing import Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
import ml_collections
import gym
import d4rl
import jax
import time
import numpy as np
import pandas as pd
from tqdm import trange
from models import CQLAgent, Off2OnAgent
from utils import PrioritizedReplayBuffer, ReplayBuffer, get_logger


def normalize_rewards(replay_buffer: ReplayBuffer, env_name: str):
    if 'maze' in env_name:
        replay_buffer.rewards = replay_buffer.rewards * 10.0 - 5.0


def eval_policy(agent, eval_env, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            action = agent.sample_action(obs, True)
            action = np.asarray(action)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


def train_and_evaluate(configs: ml_collections.ConfigDict): 
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f'cql_s{configs.seed}_{timestamp}'
    exp_info = f'# Running experiment for: {exp_name}_{configs.env_name} #'
    ckpt_dir = f"{configs.model_dir}/{configs.env_name}/{exp_name}"
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    logger = get_logger(f'logs/{configs.env_name}/{exp_name}.log')
    logger.info(f"Exp configurations:\n{configs}")

    # initialize the d4rl environment 
    env = gym.make(configs.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # load CQL ckpts
    cql_agents = []
    for cnt in [180, 200]:
        cql_agent = CQLAgent(obs_dim=obs_dim, act_dim=act_dim, max_action=max_action)
        cql_agent.load(f"saved_models/demo", cnt) 
        cql_agents.append(cql_agent)

    # initialize Off2OnRL agent
    agent = Off2OnAgent(obs_dim=obs_dim, act_dim=act_dim, max_action=max_action)

    # replay buffer
    offline_replay_buffer = ReplayBuffer(obs_dim, act_dim)
    offline_replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    online_replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer = PrioritizedReplayBuffer(obs_dim, act_dim)
    normalize_rewards(replay_buffer, configs.env_name)

    # fine-tuning
    logs = [{"step": 0, "reward": eval_policy(agent, env, configs.eval_episodes)[0]}]  # 2.38219
    for t in trange(1, configs.max_timesteps+1):
        batch = replay_buffer.sample(configs.batch_size)
        log_info = agent.update(batch)
        replay_buffer.update_priority(batch.idx, log_info["priority"])

        # Save every 1e5 steps & last 5 checkpoints
        if (t % 100000 == 0) or (t >= int(9.8e5) and t % configs.eval_freq == 0):
            agent.save(f"{ckpt_dir}", t // configs.eval_freq)

        # save some evaluate time
        if (t>int(9.5e5) and (t % configs.eval_freq == 0)) or (t<=int(9.5e5) and t % (2*configs.eval_freq) == 0):
            eval_reward, eval_time = eval_policy(agent, env, configs.eval_episodes)
            log_info.update({"step": t, "reward": eval_reward, "eval_time": eval_time, "time": (time.time()-start_time)/60})
            logs.append(log_info)
            logger.info(
                f"\n[# Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}, time: {log_info['time']:.2f}\n"
                f"\tactor_loss: {log_info['actor_loss']:.2f}, critic_loss: {log_info['critic_loss']:.2f}, alpha_loss: {log_info['alpha_loss']:.2f}\n"
                f"\tcql_loss1: {log_info['cql_loss1']:.2f}, cql_alpha_loss: {log_info['cql_alpha_loss'] if configs.with_lagrange else 0:.2f}\n"
                f"\tq1: {log_info['q1']:.2f}, target_q: {log_info['target_q']:.2f}, ood_q1: {log_info['ood_q1']:.2f}, random_q1: {log_info['random_q1']:.2f}\n"
                f"\tlogp: {log_info['logp']:.2f}, alpha: {log_info['alpha']:.2f}, cql_alpha: {log_info['cql_alpha']:.2f}\n"
            )

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{configs.log_dir}/{configs.env_name}/{exp_name}.csv")
