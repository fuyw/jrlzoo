import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import gym
import time
import ml_collections
import numpy as np
import pandas as pd
from tqdm import trange
from atari_utils import create_env
from utils import ReplayBuffer, Experience, get_logger, linear_schedule
from models import DQNAgent


def eval_policy(agent, env, eval_episodes=10):
    t1 = time.time()
    avg_reward = 0.
    act_counts = np.zeros(env.action_space.n)
    for _ in range(eval_episodes):
        obs, done = env.reset(), False  # (4, 84, 84)
        while not done:
            action = agent.sample_action(agent.state.params,
                                         np.moveaxis(obs, 0, -1)[None]).item()
            obs, reward, done, _ = env.step(action)
            act_counts[action] += 1
            avg_reward += reward
    avg_reward /= eval_episodes
    act_counts /= act_counts.sum()
    act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
    return avg_reward, act_counts, time.time() - t1


def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f"dqn_s{config.seed}_{timestamp}"
    exp_info = f'# Running experiment for: {exp_name}_{config.env_name} #'
    ckpt_dir = f"{config.ckpt_dir}/{config.env_name}/{exp_name}"
    eval_freq = config.total_timesteps // config.eval_num
    ckpt_freq = config.total_timesteps // config.ckpt_num
    explore_timesteps = config.explore_frac * config.total_timesteps
    print('#' * len(exp_info) + f'\n{exp_info}\n' + '#' * len(exp_info))

    # initialize logger
    logger = get_logger(f"{config.log_dir}/{config.env_name}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    # create envs
    env = create_env("breakout")
    num_actions = env.action_spec().num_values




    # initialize DQNAgent & Buffer
    act_dim = env.action_space.n
    agent = DQNAgent(act_dim=act_dim)
    replay_buffer = ReplayBuffer(max_size=config.buffer_size)

    # start training
    res = [{"step": 0, "eval_reward": eval_policy(agent, eval_env)[0]}]
    obs = env.reset()
    for t in trange(1, config.total_timesteps + 1):
        # greedy epsilon exploration
        epsilon = linear_schedule(start_epsilon=1.,
                                  end_epsilon=0.05,
                                  duration=explore_timesteps,
                                  t=t)

        # sample action
        if t <= config.warmup_timesteps:
            action = np.random.choice(act_dim)
        else:
            if np.random.random() < epsilon:
                action = np.random.choice(act_dim)
            else:
                context = replay_buffer.recent_obs()
                context.append(obs)
                context = np.stack(context, axis=-1)[None]  # (84, 84, 4)
                action = agent.sample_action(agent.state.params,
                                             context).item()

        # (84, 84), 0.0, False
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(Experience(obs, action, reward, done))
        obs = next_obs
        if done:
            obs = env.reset()

        # update the agent
        if (t > config.warmup_timesteps) and (t % config.train_freq == 0):
            batch = replay_buffer.sample_batch(config.batch_size)
            log_info = agent.update(batch)

        # evaluate agent
        if (t > config.warmup_timesteps) and (t % eval_freq == 0):
            eval_reward, act_counts, eval_time = eval_policy(agent, eval_env)
            logger.info(
                f"Step {t//1000}K: reward={eval_reward}, total_time={(time.time()-start_time)/60:.2f}min, "
                f"eval_time: {eval_time:.0f}s\n"
                f"\tavg_loss: {log_info['avg_loss']:.3f}, max_loss: {log_info['max_loss']:.3f}, "
                f"min_loss: {log_info['min_loss']:.3f}\n"
                f"\tavg_Q: {log_info['avg_Q']:.3f}, max_Q: {log_info['max_Q']:.3f}, "
                f"min_Q: {log_info['min_Q']:.3f}, "
                f"avg_batch_discounts: {batch.discounts.mean():.3f}\n"
                f"\tavg_target_Q: {log_info['avg_target_Q']:.3f}, max_target_Q: {log_info['max_target_Q']:.3f}, "
                f"min_target_Q: {log_info['min_target_Q']:.3f}\n"
                f"\tavg_batch_rewards: {batch.rewards.mean():.3f}, max_batch_rewards: {batch.rewards.max():.3f}, "
                f"min_batch_rewards: {batch.rewards.min():.3f}\n"
                f"\tact_counts: ({act_counts})\n"
                f"\tepsilon: {epsilon:.6f}, lr: {log_info['lr']:.6f}\n")
            log_info.update({
                "step": t,
                "eval_reward": eval_reward,
                "eval_time": eval_time
            })
            res.append(log_info)

        # save agent
        if t >= (0.9 * config.total_timesteps) and (t % ckpt_freq == 0):
            agent.save(ckpt_dir, t // ckpt_freq)

    # save logs
    # replay_buffer.save(f"{config.dataset_dir}/{exp_name}")
    df = pd.DataFrame(res).set_index("step")
    df.to_csv(f"logs/{config.env_name}/{exp_name}.csv")
