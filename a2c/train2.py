import os
import time
import ml_collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from models import A2CAgent
from utils import make_env, get_logger, add_git_info

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
    exp_prefix = f"a2c_N{config.actor_num}_L{config.rollout_len}"
    exp_name = f"s{config.seed}_{timestamp}"
    log_dir = f"logs/{exp_prefix}/{config.env_name}"
    os.makedirs(log_dir, exist_ok=True)
    exp_info = f"# Running experiment for: {exp_prefix}_{exp_name}_{config.env_name} #"
    print("#" * len(exp_info) + f"\n{exp_info}\n" + "#" * len(exp_info))

    logger = get_logger(f"{log_dir}/{exp_name}.log")
    add_git_info(config)
    logger.info(f"Exp configurations:\n{config}")

    # initialize the mujoco/dm_control environment
    envs = []
    for i in range(config.actor_num):
        env = make_env(config.env_name, config.seed + i)
        envs.append(env)
    eval_env = make_env(config.env_name, config.seed + 42)

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
    obses = np.array([env.reset() for env in envs])
    t = 0
    while t < config.max_timesteps:
        observations = np.zeros((config.rollout_len, config.actor_num, obs_dim))
        actions = np.zeros((config.rollout_len, config.actor_num, act_dim))
        rewards = np.zeros((config.rollout_len, config.actor_num))
        masks = np.zeros((config.rollout_len, config.actor_num))

        # 4 * 5 = 20
        for i in range(config.rollout_len):
            if t < config.start_timesteps:
                action = np.array([env.action_space.sample() for env in envs])
            else:
                action = agent.sample_action(obses, eval_mode=False)

            # (4, 17), (4,), (4,), (4,)
            for j in range(config.actor_num):
                next_obs, reward, done, info = envs[j].step(action[j])
                done_bool = int(done) if "TimeLimit.truncated" not in info else 0

                observations[i][j] = obses[j]
                actions[i][j] = action[j]
                rewards[i][j] = reward
                masks[i][j] = 1 - done_bool

                obses[j] = next_obs
                if done:
                    obses[j] = envs[j].reset()

        next_value = agent.get_value(obses)                            # (4,)
        returns = agent.compute_returns(next_value, rewards, masks)  # (5, 4)
        log_info = agent.update(observations, actions, returns)

        t += config.rollout_len * config.actor_num
        pbar.update(config.rollout_len * config.actor_num)

        if t % config.eval_freq:
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
