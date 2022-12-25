from typing import Any, Dict, Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
import jax
import jax.numpy as jnp
import ml_collections
import gym
import time
import numpy as np
import pandas as pd
from tqdm import trange
from models import DQNAgent, DDQNAgent
from utils import get_logger, ReplayBuffer, PrioritizedReplayBuffer

AGENTS = {"dqn": DQNAgent, "ddqn": DDQNAgent}


def eval_policy(agent: DQNAgent, env: gym.Env, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(agent.state.params, obs).item()
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward, time.time() - t1


def train_and_evaluate(configs: ml_collections.ConfigDict): 
    start_time = time.time()
    exp_name = f'{configs.algo}_s{configs.seed}_{configs.er}'
    exp_info = f'# Running experiment for: {exp_name}_{configs.env_name} #'
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    logger = get_logger(f'logs/{configs.env_name}/{configs.algo}/{exp_name}.log')
    logger.info(f"Exp configurations:\n{configs}")

    # initialize the d4rl environment 
    env = gym.make(configs.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    per = configs.er == 'per'
    agent = AGENTS[configs.algo](obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 tau=configs.tau,
                                 gamma=configs.gamma,
                                 lr=configs.lr,
                                 seed=configs.seed,
                                 per=per,
                                 per_alpha=configs.per_alpha,
                                 hidden_dims=configs.hidden_dims)

    # replay buffer
    if per:
        replay_buffer = PrioritizedReplayBuffer(obs_dim, max_size=int(1e5))
    else:
        replay_buffer = ReplayBuffer(obs_dim, max_size=int(1e5))
    logs = [{"episode":0, "reward":eval_policy(agent, env, configs.eval_episodes)[0]}]

    # start training
    episode_steps = 0
    obs, done = env.reset(), False
    for t in trange(1, configs.max_timesteps+1):
        episode_steps += 1
        if t <= configs.start_timesteps:
            action = env.action_space.sample()
        else:
            if np.random.random() <= configs.epsilon:
                action = env.action_space.sample()
            else:
                action = agent.sample_action(agent.state.params, obs).item()

        next_obs, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_steps < env._max_episode_steps else 0
        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs

        if t > configs.start_timesteps:
            batch = replay_buffer.sample(configs.batch_size)
            log_info = agent.update(batch)
            if per:
                replay_buffer.update_priority(batch.idx, log_info["priority"])

            if (t % configs.eval_freq == 0):
                eval_reward, eval_time = eval_policy(agent, env, configs.eval_episodes)
                log_str = f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}, " +\
                        f"time: {(time.time()-start_time)/60:.2f}\n" +\
                        f"\tloss: {log_info['loss']:.3f}, Q: {log_info['Q']:.3f}, target_Q: {log_info['target_Q']:.3f}\n" +\
                        f"\tbatch_rewards: {log_info['batch_rewards']:.0f}, batch_actions: {log_info['batch_actions'].sum():.0f}\n"
                if per: log_str += f", batch_weights: {batch.weights.mean():.3f}"
                logger.info(log_str) 
                logs.append(dict(episode=t, reward=eval_reward, loss=log_info['loss'].item(), Q=log_info['Q'].item()))

        if done:
            obs, done = env.reset(), False
            episode_steps = 0

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{configs.log_dir}/{configs.env_name}/{configs.algo}/{exp_name}.csv")
