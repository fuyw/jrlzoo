import os
import time
import ml_collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from models import A2CAgent
from utils import SubprocVecEnv, env_fn, get_logger, add_git_info

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"


def eval_policy(agent, env, eval_episodes=10):
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(obs, eval_mode=True)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward, time.time() - t1


def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # logging
    exp_prefix = f"a2c_N{config.actor_num}_L{config.rollout_len}" +\
        f"_{config.max_timesteps//1000000}M"
    exp_name = f"s{config.seed}_{timestamp}"
    log_dir = f"logs/{exp_prefix}/{config.env_name}"
    os.makedirs(log_dir, exist_ok=True)
    exp_info = f"# Running experiment for: {exp_prefix}_{exp_name}_{config.env_name} #"
    print("#" * len(exp_info) + f"\n{exp_info}\n" + "#" * len(exp_info))

    logger = get_logger(f"{log_dir}/{exp_name}.log")
    add_git_info(config)
    logger.info(f"Exp configurations:\n{config}")

    # initialize envs
    envs = [env_fn(config.env_name, seed=i) for i in range(config.actor_num)]
    envs = SubprocVecEnv(envs)
    eval_env = env_fn(config.env_name, seed=config.seed*100)()

    # initialize the agent
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    agent = A2CAgent(obs_dim=obs_dim,
                     act_dim=act_dim,
                     seed=config.seed)

    # eval the untrained agent
    logs = [{
        "step": 0,
        "reward": eval_policy(agent, eval_env, config.eval_episodes)[0]
    }]

    # start training
    pbar = tqdm(total=config.max_timesteps)
    t, obs = 0, envs.reset()
    while t < config.max_timesteps:
        observations = []
        actions = []
        values = []
        rewards = []
        masks = []

        # 4 * 5 = 20
        for _ in range(config.rollout_len):
            # mean_action, sampled_action, logp, value
            action = agent.sample_action(obs, eval_mode=False)

            # (4, 17), (4,), (4,), (4,)
            next_obs, reward, done, info = envs.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            masks.append(1 - done)

            obs = next_obs

        next_value = agent.get_value(next_obs)                       # (4,)
        returns = agent.compute_returns(next_value, rewards, masks)  # (5, 4)
        observations = np.array(observations)                        # (5, 4, 15)
        actions = np.array(actions)                                  # (5, 4, 4)

        log_info = agent.update(observations, actions, returns)

        t += config.rollout_len * config.actor_num
        pbar.update(config.rollout_len * config.actor_num)

        if t % config.eval_freq == 0:
            eval_reward, eval_time = eval_policy(agent,
                                                 eval_env,
                                                 config.eval_episodes)
            log_info.update({"step": t, "reward": eval_reward})
            logs.append(log_info)
            logger.info(
                f"\n[#Step {t//1000}K] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}\n"
                f"\tpi_loss: {log_info['actor_loss']:.3f}, "
                f"q_loss: {log_info['critic_loss']:.3f}, "
                f"e_loss: {log_info['entropy_loss']:.3f}\n"
            )

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(
        f"logs/{exp_prefix}/{config.env_name}/{exp_name}.csv")
