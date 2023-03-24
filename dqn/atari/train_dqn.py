import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import time

import gym
import numpy as np
import pandas as pd
from tqdm import trange

from atari_wrappers import wrap_deepmind
from models import DQN2Agent
from utils import Experience, ReplayBuffer, get_logger, linear_schedule


###################
# Utils Functions #
###################
def eval_policy(agent, env):
    t1 = time.time()
    obs = env.reset()
    act_counts = np.zeros(env.action_space.n)
    while not env.get_real_done():
        action = agent.sample(obs[None])
        act_counts[action] += 1
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    act_counts /= act_counts.sum()
    act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
    return np.mean(env.get_eval_rewards()), act_counts, time.time() - t1


#################
# Main Function #
#################
def train_and_evaluate(config):
    # general setting
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f"dqn2_s{config.seed}_{timestamp}"
    exp_info = f'# Running experiment for: {exp_name}_{config.env_name} #'
    eval_freq = config.total_timesteps // config.eval_num
    ckpt_freq = config.total_timesteps // config.ckpt_num
    ckpt_dir = f"{config.model_dir}/{config.env_name}"

    print('#' * len(exp_info) + f'\n{exp_info}\n' + '#' * len(exp_info))
    logger = get_logger(f"logs/{config.env_name}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    # make environments
    env = gym.make(f"{config.env_name}NoFrameskip-v4")
    env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")
    eval_env = gym.make(f"{config.env_name}NoFrameskip-v4")
    eval_env = wrap_deepmind(eval_env, dim=84, obs_format="NHWC", test=True)
    act_dim = env.action_space.n

    # initialize DQN agent
    agent = DQN2Agent(act_dim=act_dim, seed=config.seed)
    agent.load(f"saved_models/cql/{config.env_name}", 1)

    # create the replay buffer
    replay_buffer = ReplayBuffer(max_size=1e6)

    # start training
    obs = env.reset()   # (84, 84)
    fps_t1 = None
    res = [{"step": 0, "eval_reward": eval_policy(agent, eval_env)[0]}]
    logger.info(f"Step {0}K [{0.:.1f}%]: reward={res[0]['eval_reward']:.1f}\n")

    # pre-train
    # check_reward = 0
    # while check_reward < 1000:
    #     for t in trange(1, 1+config.total_timesteps):
    #         epsilon = linear_schedule(0.2, 0.05, config.explore_frac*config.total_timesteps, t)
    #         if t <= config.warmup_timesteps:
    #             action = np.random.choice(act_dim)
    #         else:
    #             if np.random.random() < epsilon:
    #                 action = np.random.choice(act_dim)
    #             else:
    #                 context = replay_buffer.recent_obs()
    #                 context.append(obs)
    #                 context = np.stack(context, axis=-1)[None]
    #                 action = agent.sample(context)

    #         # interact with the environment
    #         next_obs, reward, done, _ = env.step(action)
    #         replay_buffer.add(Experience(obs, action, reward, done))
    #         obs = next_obs
    #         if done:
    #             obs = env.reset()

    #         # update the agent
    #         if (t > 10_000) and (t % config.train_freq == 0):
    #             batch = replay_buffer.sample_batch(config.batch_size)
    #             _ = agent.update(batch)
    #             if t % config.update_target_freq == 0:
    #                 agent.sync_target_network()

    #         if (t > config.warmup_timesteps) and (t % 100_000 == 0):
    #             check_reward, act_counts, eval_time = eval_policy(agent, eval_env)
    #             logger.info(f"Step {t}K [{t/config.total_timesteps*100.:.1f}%]: reward={check_reward:.1f}\n")
    # res = [{"step": 0, "eval_reward": check_reward}]
    # logger.info(f"Step {0}K [{0.:.1f}%]: reward={check_reward:.1f}\n")

    # fine-tuning
    for t in trange(1, 1+config.total_timesteps):
        epsilon = linear_schedule(0.2, 0.05, config.explore_frac*config.total_timesteps, t)
        if t <= config.warmup_timesteps:
            action = np.random.choice(act_dim)
        else:
            if np.random.random() < epsilon:
                action = np.random.choice(act_dim)
            else:
                context = replay_buffer.recent_obs()
                context.append(obs)
                context = np.stack(context, axis=-1)[None]
                action = agent.sample(context)

        # interact with the environment
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(Experience(obs, action, reward, done))
        obs = next_obs
        if done:
            obs = env.reset()

        # update the agent
        if (t > config.warmup_timesteps) and (t % config.train_freq == 0):
            batch = replay_buffer.sample_batch(config.batch_size)
            log_info = agent.update(batch)
            if t % config.update_target_freq == 0:
                agent.sync_target_network()

        # evaluate the agent
        if (t > config.warmup_timesteps) and (t % eval_freq == 0):
            fps_t2, fps = time.time(), np.nan
            if fps_t1 is not None:
                log_info["fps"] = fps = eval_freq/(fps_t2 - fps_t1)
            eval_reward, act_counts, eval_time = eval_policy(agent, eval_env)
            log_info.update({"step": t,"eval_reward": eval_reward, "eval_time": eval_time, 'total_time': (time.time()-start_time)/60})
            res.append(log_info)
            logger.info(f"Step {t//1000}K [{t/config.total_timesteps*100.:.1f}%]: reward={eval_reward:.1f}\n"
                        f"\teval_time={eval_time:.1f}s, total_time={log_info['total_time']:.1f}min, fps={fps:.0f}\n"
                        f"\tavg_loss: {log_info['avg_loss']:.2f}, avg_Q: {log_info['avg_Q']:.2f}, avg_target_Q: {log_info['avg_target_Q']:.2f}\n"
                        f"\tavg_batch_rewards: {batch.rewards.mean():.3f}, avg_batch_discounts: {batch.discounts.mean():.3f}\n"
                        f"\tact_counts: ({act_counts}), epsilon={epsilon:.3f}\n")
            fps_t1 = time.time()

    res_df = pd.DataFrame(res)
    res_df.to_csv(f"logs/{config.env_name}/{exp_name}.csv")
