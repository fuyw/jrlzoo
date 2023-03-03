"""Online SAC Agent"""
from typing import Tuple
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import ml_collections
import gymnasium as gym
import time
import pandas as pd
from tqdm import trange
from models import SACAgent
from utils import ReplayBuffer, get_logger
# from gym_utils import make_env


def eval_policy(agent: SACAgent,
                env: gym.Env,
                eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward, avg_step = 0., 0.
    for _ in range(eval_episodes):
        (obs, _), done, truncated = env.reset(), False, False
        while (not done and not truncated):
            action = agent.sample_action(obs, True)
            obs, reward, done, truncated, _ = env.step(action)
            avg_reward += reward
            avg_step += 1
    avg_reward /= eval_episodes
    avg_step /= eval_episodes
    return avg_reward, avg_step, time.time() - t1


def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"sac_s{config.seed}_{timestamp}"
    exp_info = f"# Running experiment for: {exp_name}_{config.env_name} #"
    ckpt_dir = f"{config.model_dir}/{config.env_name.lower()}/{exp_name}"
    print("#" * len(exp_info) + f"\n{exp_info}\n" + "#" * len(exp_info))

    logger = get_logger(f"logs/{config.env_name.lower()}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    env = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # SAC agent
    agent = SACAgent(obs_dim=obs_dim,
                     act_dim=act_dim,
                     max_action=max_action,
                     seed=config.seed,
                     tau=config.tau,
                     gamma=config.gamma,
                     lr=config.lr,
                     hidden_dims=config.hidden_dims,
                     initializer=config.initializer)

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    logs = [{"step": 0, "reward": eval_policy(agent, eval_env, config.eval_episodes)[0]}]

    obs, _  = env.reset()
    for t in trange(1, config.max_timesteps+1):
        if t <= config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.sample_action(obs)

        next_obs, reward, done, truncated, info = env.step(action)
        done_bool = float(done) if not truncated else 0

        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs

        if done or truncated:
            (obs, _), done, truncated = env.reset(), False, False

        if t > config.start_timesteps:
            batch = replay_buffer.sample(config.batch_size)
            log_info = agent.update(batch)

        if ((t>int(9.5e5) and (t % config.eval_freq == 0)) or (t<=int(9.5e5) and t % (2*config.eval_freq) == 0)):
            eval_reward, eval_step, eval_time = eval_policy(agent, eval_env, config.eval_episodes)
            if t > config.start_timesteps:
                log_info.update({
                    "step": t,
                    "reward": eval_reward,
                    "eval_time": eval_time,
                    "eval_step": eval_step,
                    "time": (time.time() - start_time) / 60
                })
                logger.info(
                    f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}, eval_step: {eval_step:.2f}, time: {log_info['time']:.2f}\n"
                    f"\tactor_loss: {log_info['actor_loss']:.3f}, critic_loss: {log_info['critic_loss']:.3f}, alpha_loss: {log_info['alpha_loss']:.3f}\n"
                    f"\tq1: {log_info['q1']:.2f}, target_q: {log_info['target_q']:.2f}, logp: {log_info['logp']:.3f}, alpha: {log_info['alpha']:.3f}\n"
                    f"\tbatch_reward: {batch.rewards.mean():.3f}, batch_reward_min: {batch.rewards.min():.3f}, batch_reward_max: {batch.rewards.max():.3f}\n"
                    f"\tbatch_discount: {batch.discounts.mean():.3f}, batch_discount_min: {batch.discounts.min():.3f}, batch_discount_max: {batch.discounts.max():.3f}\n"
                    f"\tbuffer_size: {replay_buffer.size//1000}, buffer_ptr: {replay_buffer.ptr//1000}\n"
                )
                logs.append(log_info)
            else:
                logs.append({"step": t, "reward": eval_reward})
                logger.info(
                    f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}\n"
                )

        # Save checkpoints
        if t % config.ckpt_freq == 0:
            agent.save(f"{ckpt_dir}", t // config.ckpt_freq)

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(
        f"{config.log_dir}/{config.env_name.lower()}/{exp_name}.csv")

    # Save buffer
    # if config.save_buffer:
    #     buffer_dir = f"saved_buffers/{config.env_name}"
    #     os.makedirs(buffer_dir, exist_ok=True)
    #     replay_buffer.save(f"{buffer_dir}/{config.env_name}")
