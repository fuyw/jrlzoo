import os
import time

import imageio
import jax
import numpy as np
import pandas as pd
from ml_collections import ConfigDict
from tqdm import trange

from models import DDPGAgent, SACAgent
from utils import (Batch, HERBuffer,  add_git_info, get_logger,
                   make_env, target_update)


def eval_agent(agent,
               eval_env,
               eval_episodes: int = 100,
               save_video: bool = False):
    avg_reward = 0
    avg_step = 0
    avg_success_rate = 0
    frames = []
    for i in range(1, 1+eval_episodes):
        observation, _ = eval_env.reset()
        if save_video and i == eval_episodes:
            frames.append(eval_env.render())
        while True:
            action = agent.sample_action(observation["observation"],
                                         observation["desired_goal"])
            observation, reward, terminal, truncation, info = eval_env.step(action)
            if save_video and i == eval_episodes:
                frames.append(eval_env.render())
            avg_reward += reward
            avg_step += 1
            if terminal or truncation:
                avg_success_rate += info["is_success"]
                break
    avg_reward /= eval_episodes
    avg_step /= eval_episodes
    avg_success_rate /= eval_episodes
    return avg_success_rate, avg_reward, avg_step, frames


def run_old(config):
    # logging dir
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_param = f"N{config.num_envs}_L{config.max_episode_steps}"
    exp_name = f"s{config.seed}_{timestamp}"
    exp_prefix = f"{config.model}/{config.env_name}/{exp_param}"
    os.makedirs(f"logs/{exp_prefix}", exist_ok=True)
    ckpt_dir = f"{os.getcwd()}/saved_models/{exp_prefix}/{exp_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if config.save_video:
        os.makedirs(f"saved_videos/{exp_prefix}/{exp_name}", exist_ok=True)
    logger = get_logger(f"logs/{exp_prefix}/{exp_name}.log")
    add_git_info(config)
    logger.info(f"Config:\n{config}\n")

    # random seed
    np.random.seed(config.seed)

    # initialize vectorized env
    N = config.num_envs
    envs = make_env(config.env_name,
                    num_envs=N,
                    max_episode_steps=config.max_episode_steps)
    eval_env = make_env(config.env_name,
                        num_envs=1,
                        max_episode_steps=config.max_episode_steps,
                        render_mode="rgb_array")
    act_dim = eval_env.action_space.shape[0]
    obs_dim = eval_env.observation_space["observation"].shape[0]
    goal_dim = eval_env.observation_space["desired_goal"].shape[0]
    max_action = eval_env.action_space.high[0]  # 1.0
    traj_len = eval_env._max_episode_steps  # 50

    # initialize RL agent
    if config.model == "sac":
        agent = SACAgent(obs_dim=obs_dim,
                         act_dim=act_dim,
                         goal_dim=goal_dim,
                         seed=config.seed,
                         lr=config.lr,
                         hidden_dims=config.hidden_dims,
                         ckpt_dir=ckpt_dir)
    elif config.model == "ddpg":
        agent = DDPGAgent(obs_dim=obs_dim,
                          act_dim=act_dim,
                          goal_dim=goal_dim,
                          seed=config.seed,
                          lr=config.lr,
                          hidden_dims=config.hidden_dims,
                          ckpt_dir=ckpt_dir)

    # replay buffer
    buffer = HERBuffer(obs_dim=obs_dim,
                       act_dim=act_dim,
                       goal_dim=goal_dim,
                       replay_k=config.replay_k,
                       max_size=config.max_size,
                       traj_len=traj_len,
                       reward_fn=eval_env.unwrapped.compute_reward)

    # trajectory data
    traj_observations = np.zeros((N, traj_len+1, obs_dim),
                                 dtype=np.float32)
    traj_achieved_goals = np.zeros((N, traj_len+1, goal_dim),
                                   dtype=np.float32)
    traj_goals = np.zeros((N, traj_len, goal_dim),
                          dtype=np.float32)
    traj_actions = np.zeros((N, traj_len, act_dim),
                            dtype=np.float32)
    traj_dones = np.zeros((N, traj_len), dtype=bool)
    traj_ptr = 0

    # reset the environment
    _ = eval_env.reset(seed=config.seed+123)
    observations, _ = envs.reset(seed=config.seed)
    traj_observations[:, 0, :] = observations["observation"]
    traj_achieved_goals[:, 0, :] = observations["achieved_goal"]

    logs = []
    for t in trange(N, N+config.max_timesteps, N):
        if t <= config.start_timesteps:
            actions = envs.action_space.sample()
        else:
            if config.model == "ddpg":
                noises = np.random.normal(0, max_action * config.expl_noise,
                                          size=(N, act_dim))
                actions = agent.sample_action(observations["observation"],
                                              observations["desired_goal"])
                actions = (actions + noises).clip(-max_action, max_action)
            elif config.model == "sac":
                actions = agent.sample_action(observations["observation"],
                                              observations["desired_goal"])

        # interact with the environment
        observations, rewards, terminals, truncations, _ = envs.step(actions)

        # save trajectory data
        traj_observations[:, traj_ptr+1, :] = observations["observation"]
        traj_achieved_goals[:, traj_ptr+1, :] = observations["achieved_goal"]
        traj_goals[:, traj_ptr, :] = observations["desired_goal"]
        traj_actions[:, traj_ptr, :] = actions
        traj_dones[:, traj_ptr] = terminals
        traj_ptr += 1

        if traj_ptr == traj_len:
            traj_ptr = 0
            buffer.add(traj_observations,
                       traj_achieved_goals,
                       traj_goals,
                       traj_actions,
                       traj_dones)
            observations, _ = envs.reset()
            traj_observations[:, 0, :] = observations["observation"]
            traj_achieved_goals[:, 0, :] = observations["achieved_goal"]

        # train the agent
        if t > config.start_timesteps:
            for _ in range(N):
                batch = buffer.sample(config.batch_size)
                log_info = agent.update(batch) 
 
        # save model checkpoint
        if t % config.ckpt_freq == 0:
            agent.save(t//config.ckpt_freq)

        # evaluate the agent
        if t % config.eval_freq == 0:
            (success_rate,
             eval_reward,
             eval_step,
             eval_frames) = eval_agent(agent,
                                       eval_env,
                                       config.eval_episodes,
                                       save_video=config.save_video)
            log_info.update({
                "t": t,
                "eval_reward": eval_reward,
                "eval_step": eval_step,
                "success_rate": success_rate,
                "time": (time.time() - start_time) / 60,
            })
            logger.info(
                f"\n[#T {t//1000}] success_rate: {success_rate:.2f}, time: {log_info['time']:.2f}\n"
                f"\teval_step: {eval_step:.2f}, eval_reward: {eval_reward:.2f}\n"
                f"\tcritic_loss: {log_info['critic_loss']:.2f}, actor_loss: {log_info['actor_loss']:.2f}\n"
                f"\tq1: {log_info['q1']:.2f}, max_q1: {log_info['max_q1']:.2f}, min_q1: {log_info['min_q1']:.2f}\n"
                f"\tgoal: {', '.join([f'{i:.3f}' for i in observations['desired_goal'].sum(-1)[:5]])}\n"
            )
            logs.append(log_info)
            if eval_frames:
                imageio.mimsave(f"saved_videos/{exp_prefix}/{exp_name}/{t//config.eval_freq}.mp4", eval_frames)

    # close environments
    envs.close()
    eval_env.close()

    # save log
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"logs/{exp_prefix}/{exp_name}.csv")


def run(config):
    # logging dir
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_param = f"N{config.num_envs}_L{config.max_episode_steps}"
    exp_name = f"s{config.seed}_{timestamp}"
    exp_prefix = f"{config.model}/{config.env_name}/{exp_param}"
    os.makedirs(f"logs/{exp_prefix}", exist_ok=True)
    ckpt_dir = f"{os.getcwd()}/saved_models/{exp_prefix}/{exp_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if config.save_video:
        os.makedirs(f"saved_videos/{exp_prefix}/{exp_name}", exist_ok=True)
    logger = get_logger(f"logs/{exp_prefix}/{exp_name}.log")
    add_git_info(config)
    logger.info(f"Config:\n{config}\n")

    # random seed
    np.random.seed(config.seed)

    # initialize vectorized env
    N = config.num_envs
    envs = make_env(config.env_name,
                    num_envs=N,
                    max_episode_steps=config.max_episode_steps)
    eval_env = make_env(config.env_name,
                        num_envs=1,
                        max_episode_steps=config.max_episode_steps,
                        render_mode="rgb_array")
    act_dim = eval_env.action_space.shape[0]
    obs_dim = eval_env.observation_space["observation"].shape[0]
    goal_dim = eval_env.observation_space["desired_goal"].shape[0]
    max_action = eval_env.action_space.high[0]  # 1.0
    traj_len = eval_env._max_episode_steps  # 50

    # initialize RL agent
    if config.model == "sac":
        agent = SACAgent(obs_dim=obs_dim,
                         act_dim=act_dim,
                         goal_dim=goal_dim,
                         seed=config.seed,
                         lr=config.lr,
                         hidden_dims=config.hidden_dims,
                         ckpt_dir=ckpt_dir)
    elif config.model == "ddpg":
        agent = DDPGAgent(obs_dim=obs_dim,
                          act_dim=act_dim,
                          goal_dim=goal_dim,
                          seed=config.seed,
                          lr=config.lr,
                          hidden_dims=config.hidden_dims,
                          ckpt_dir=ckpt_dir)

    # replay buffer
    buffer = HERBuffer(obs_dim=obs_dim,
                       act_dim=act_dim,
                       goal_dim=goal_dim,
                       replay_k=config.replay_k,
                       max_size=config.max_size,
                       traj_len=traj_len,
                       reward_fn=eval_env.unwrapped.compute_reward)

    # trajectory data
    traj_observations = np.zeros((N, traj_len+1, obs_dim),
                                 dtype=np.float32)
    traj_achieved_goals = np.zeros((N, traj_len+1, goal_dim),
                                   dtype=np.float32)
    traj_goals = np.zeros((N, traj_len, goal_dim),
                          dtype=np.float32)
    traj_actions = np.zeros((N, traj_len, act_dim),
                            dtype=np.float32)
    traj_dones = np.zeros((N, traj_len), dtype=bool)
    traj_ptr = 0

    # reset the environment
    _ = eval_env.reset(seed=config.seed+123)
    observations, _ = envs.reset(seed=config.seed)
    traj_observations[:, 0, :] = observations["observation"]
    traj_achieved_goals[:, 0, :] = observations["achieved_goal"]

    logs = []

    t = 0
    for epoch in trange(50):
        traj_cnt = 0
        while traj_cnt < 100:
            if t <= config.start_timesteps:
                actions = envs.action_space.sample()
            else:
                if config.model == "ddpg":
                    noises = np.random.normal(0, max_action * config.expl_noise,
                                            size=(N, act_dim))
                    actions = agent.sample_action(observations["observation"],
                                                observations["desired_goal"])
                    actions = (actions + noises).clip(-max_action, max_action)
                elif config.model == "sac":
                    actions = agent.sample_action(observations["observation"],
                                                observations["desired_goal"])

            # interact with the environment
            observations, rewards, terminals, truncations, _ = envs.step(actions)
            t += N

            # save trajectory data
            traj_observations[:, traj_ptr+1, :] = observations["observation"]
            traj_achieved_goals[:, traj_ptr+1, :] = observations["achieved_goal"]
            traj_goals[:, traj_ptr, :] = observations["desired_goal"]
            traj_actions[:, traj_ptr, :] = actions
            traj_dones[:, traj_ptr] = terminals
            traj_ptr += 1

            if traj_ptr == traj_len:
                traj_ptr = 0
                traj_cnt += 1
                buffer.add(traj_observations,
                           traj_achieved_goals,
                           traj_goals,
                           traj_actions,
                           traj_dones)
                observations, _ = envs.reset()
                traj_observations[:, 0, :] = observations["observation"]
                traj_achieved_goals[:, 0, :] = observations["achieved_goal"]

                # train the agent
                if t > config.start_timesteps:
                    for _ in range(20):
                        batch = buffer.sample(config.batch_size)
                        log_info = agent.update(batch) 
    
        # save model checkpoint
        if (epoch + 1) % 10 == 0:
            agent.save((epoch+1)//10)

        # evaluate the agent
        # if t % config.eval_freq == 0:
        (success_rate,
         eval_reward,
         eval_step,
         eval_frames) = eval_agent(agent,
                                   eval_env,
                                   config.eval_episodes,
                                   save_video=config.save_video)
        log_info.update({
            "t": t,
            "epoch": epoch,
            "eval_reward": eval_reward,
            "eval_step": eval_step,
            "success_rate": success_rate,
            "time": (time.time() - start_time) / 60,
        })
        logger.info(
            f"\n[E {epoch+1}][T {t//1000}] success_rate: {success_rate:.2f}, time: {log_info['time']:.2f}\n"
            f"\teval_step: {eval_step:.0f}, eval_reward: {eval_reward:.2f}\n"
            f"\tcritic_loss: {log_info['critic_loss']:.2f}, actor_loss: {log_info['actor_loss']:.2f}\n"
            f"\tq1: {log_info['q1']:.2f}, max_q1: {log_info['max_q1']:.2f}, min_q1: {log_info['min_q1']:.2f}\n"
            f"\tgoal: {', '.join([f'{i:.3f}' for i in observations['desired_goal'].sum(-1)[:5]])}\n"
        )
        logs.append(log_info)
        if eval_frames:
            imageio.mimsave(f"saved_videos/{exp_prefix}/{exp_name}/{t//config.eval_freq}.mp4", eval_frames)

    # close environments
    envs.close()
    eval_env.close()

    # save log
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"logs/{exp_prefix}/{exp_name}.csv")
