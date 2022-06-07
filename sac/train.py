"""Online SAC Agent"""
from typing import Tuple
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import ml_collections
import gym
import time
import pandas as pd
from tqdm import trange
from models import SACAgent
from utils import ReplayBuffer, get_logger
from gym_utils import make_env


def eval_policy(agent: SACAgent,
                env: gym.Env,
                eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            # dummy rng in evaluation step
            agent.rng, action = agent.sample_action(agent.actor_state.params, agent.rng, obs, True)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward, time.time() - t1


def train_and_evaluate(configs: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f'sac_s{configs.seed}_ups{configs.updates_per_step}_{timestamp}'
    exp_info = f'# Running experiment for: {exp_name}_{configs.env_name} #'
    ckpt_dir = f"{configs.model_dir}/{configs.env_name.lower()}/ups{configs.updates_per_step}/{exp_name}"
    print('#' * len(exp_info) + f'\n{exp_info}\n' + '#' * len(exp_info))

    logger = get_logger(f'logs/{configs.env_name.lower()}/ups{configs.updates_per_step}/{exp_name}.log')
    logger.info(f"Exp configurations:\n{configs}")

    # env = gym.make(configs.env_name)
    env = make_env(configs.env_name, configs.seed)
    eval_env = make_env(configs.env_name, configs.seed + 42)
    # np.random.seed(FLAGS.seed)
    # random.seed(FLAGS.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # SAC agent
    agent = SACAgent(obs_dim=obs_dim,
                     act_dim=act_dim,
                     max_action=max_action,
                     seed=configs.seed,
                     tau=configs.tau,
                     gamma=configs.gamma,
                     lr=configs.lr,
                     hidden_dims=configs.hidden_dims,
                     initializer=configs.initializer)

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    logs = [{
        "step": 0,
        "reward": eval_policy(agent, eval_env, configs.eval_episodes)[0]
    }]

    obs, done = env.reset(), False
    for t in trange(1, configs.max_timesteps + 1):
        if t <= configs.start_timesteps:
            action = env.action_space.sample()
        else:
            agent.rng, action = agent.sample_action(agent.actor_state.params,
                                                    agent.rng, obs)

        next_obs, reward, done, info = env.step(action)
        done_bool = float(done) if 'TimeLimit.truncated' not in info else 0

        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs

        if t > configs.start_timesteps:
            for _ in range(configs.updates_per_step):
                batch = replay_buffer.sample(configs.batch_size)
                log_info = agent.update(batch)

        if done:
            obs, done = env.reset(), False

        if t % configs.eval_freq == 0:
            eval_reward, eval_time = eval_policy(agent, eval_env, configs.eval_episodes)
            if t > configs.start_timesteps:
                log_info.update({
                    "step": t,
                    "reward": eval_reward,
                    "eval_time": eval_time,
                    "time": (time.time() - start_time) / 60
                })
                logger.info(
                    f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}, time: {log_info['time']:.2f}\n"

                    f"\tcritic_loss1: {log_info['critic_loss1']:.3f}, critic_loss1_max: {log_info['critic_loss1_max']:.3f}, critic_loss1_min: {log_info['critic_loss1_min']:.3f}, critic_loss1_std: {log_info['critic_loss1_std']:.3f}\n"
                    f"\tcritic_loss2: {log_info['critic_loss2']:.3f}, critic_loss2_max: {log_info['critic_loss2_max']:.3f}, critic_loss2_min: {log_info['critic_loss2_min']:.3f}, critic_loss2_std: {log_info['critic_loss2_std']:.3f}\n"

                    f"\tactor_loss: {log_info['actor_loss']:.3f}, actor_loss_max: {log_info['actor_loss_max']:.3f}, actor_loss_min: {log_info['actor_loss_min']:.3f}, actor_loss_std: {log_info['actor_loss_std']:.3f}\n"

                    f"\talpha_loss: {log_info['alpha_loss']:.3f}, alpha_loss_max: {log_info['alpha_loss_max']:.3f}, alpha_loss_min: {log_info['alpha_loss_min']:.3f}, alpha_loss_std: {log_info['alpha_loss_std']:.3f}\n"

                    f"\tq1: {log_info['q1']:.3f}, q1_max: {log_info['q1_max']:.3f}, q1_min: {log_info['q1_min']:.3f}, q1_std: {log_info['q1_std']:.3f}\n"
                    f"\tq2: {log_info['q2']:.3f}, q2_max: {log_info['q2_max']:.3f}, q2_min: {log_info['q2_min']:.3f}, q2_std: {log_info['q2_std']:.3f}\n"
                    f"\ttarget_q: {log_info['target_q']:.3f}, target_q_max: {log_info['target_q_max']:.3f}, target_q_min: {log_info['target_q_min']:.3f}, target_q_std: {log_info['target_q_std']:.3f}\n"

                    f"\tlogp: {log_info['logp']:.3f}, alpha: {log_info['alpha']:.3f}\n"
                    f"\tcritic_param_norm: {log_info['critic_param_norm']:.3f}, actor_param_norm: {log_info['actor_param_norm']:.3f}\n"
                )
                logs.append(log_info)
            else:
                logs.append({"step": t, "reward": eval_reward})
                logger.info(
                    f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}\n"
                )

        # Save checkpoints
        if t % configs.ckpt_freq == 0:
            agent.save(f"{ckpt_dir}", t // configs.ckpt_freq)

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(
        f"{configs.log_dir}/{configs.env_name.lower()}/ups{configs.updates_per_step}/{exp_name}.csv")
