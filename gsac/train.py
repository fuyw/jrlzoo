"""Online SAC Agent"""
from typing import Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import ml_collections
import gym
import random
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import trange
from models import SACAgent
from utils import ReplayBuffer, get_logger
from gym_utils import make_env


##############
# CheckPoint #
##############
class CheckPoint:
    def __init__(self, state, future_reward, ep_len,
                 observation, epoch, old_reward, future_steps_num):
        self.state = state
        self.future_reward = future_reward
        self.future_steps_num = future_steps_num
        self.ep_len = ep_len
        self.observation = observation
        self.epoch = epoch
        self.old_reward = old_reward
    
    def get_save_score(self):
        return self.future_reward

    def get_sample_score(self, agent):
        action = agent.sample(self.observation)        
        q = agent.get_q(self.observation, action)
        return -q

    def get_avg_future_reward(self):
        return self.future_reward / self.future_steps_num


def sample_checkpoint_idx_by_rule(agent, check_points, sample_num=1, best_threshold=2.0):
    n = 0
    best_idx = -1
    best_score = 0
    old_ret = 0
    for _ in range(3 * sample_num):
        idx = np.random.randint(0, len(check_points)-1)
        score = check_points[idx].get_sample_score(agent)
        avg_future_reward = check_points[idx].get_avg_future_reward


def update_check_points(trajs, check_points, old_idx, epoch, future_reward_step_num=50):
    # trajs[0] = [state, obs, reward, ep_len]
    rewards = [x[2] for x in trajs]
    cumulative_rewards = [0] * (len(rewards)+1)
    for i in range(len(rewards)):
        cumulative_rewards[i+1] = cumulative_rewards[i] + rewards[i]
    total_reward = cumulative_rewards[-1]
    old_reward = 0
    if old_idx >= 0 and len(trajs) >= future_reward_step_num:
        old_reward = check_points[old_idx].old_reward
        new_future_reward = cumulative_rewards



##################
# Utils Function #
##################
def eval_policy(agent: SACAgent,
                env: gym.Env,
                eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward, avg_step = 0., 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            avg_step += 1
            action = agent.sample_action(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    avg_step /= eval_episodes
    return avg_reward, avg_step, time.time() - t1


def save_state(env):
    return env.sim.get_state()


def restore_state(env, state):
    env.reset()
    env.sim.set_state(state)
    env.sim.forward()
    return env.get_obs()


def get_random_init_state(trajs, extra_step_num):
    while True:
        tmp = trajs[np.random.randint(0, len(trajs)-1)]
        if tmp[0] + extra_step_num < 990:
            return tmp


def run_extra_steps(env, agent, step_num, obs, ep_len, max_ep_len=1000):
    total_reward = 0
    cnt = 0
    for i in range(step_num):
        cnt += 1
        obs, reward, done, _ = env.step(agent.sample_action(obs))
        total_reward += reward
        ep_len += 1
        if done or (ep_len == max_ep_len):
            break
    return done, total_reward, cnt


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"gsac_s{config.seed}_{timestamp}"
    exp_info = f"# Running experiment for: {exp_name}_{config.env_name} #"
    ckpt_dir = f"{config.model_dir}/{config.env_name.lower()}/{exp_name}"
    print("#"*len(exp_info) + f"\n{exp_info}\n" + "#"*len(exp_info))

    logger = get_logger(f"logs/{config.env_name.lower()}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    # initialize the mujoco/dm_control environment
    if "v2" in config.env_name:
        env = gym.make(config.env_name)
        eval_env = gym.make(config.env_name)
    else:
        env = make_env(config.env_name, config.seed)
        eval_env = make_env(config.env_name, config.seed + 42)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    np.random.seed(config.seed)
    random.seed(config.seed)

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
    logs = [{
        "step": 0,
        "reward": eval_policy(agent, eval_env, config.eval_episodes)[0]
    }]

    # initial state
    obs = env.reset()
    ep_len = 0

    # record trajectory information
    max_step_remain = 1000
    trajs = [(save_state(env), obs, 0, ep_len)]
    old_id = -1

    # checkpoints
    check_points = []

    # start training
    for t in trange(1, config.max_timesteps + 1):
        if t <= config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.sample_action(obs)

        next_obs, reward, done, info = env.step(action)
        done_bool = float(done) if "TimeLimit.truncated" not in info else 0

        replay_buffer.add(obs, action, next_obs, reward, done_bool)
        obs = next_obs

        # record trajectory information
        max_step_remain -= 1
        ep_len += 1
        trajs.append((save_state(env), obs, reward, ep_len))

        if t > config.start_timesteps:
            batch = replay_buffer.sample(config.batch_size)
            log_info = agent.update(batch)

        if done or max_step_remain == 0:
            obs, done = env.reset(), False
            max_step_remain = 1000
            trajs = [(save_state(env), obs, 0, ep_len)]
            old_id = -1

            # reset from old trajectories
            ratio = 1000. / (1000. + (1./config.traj_sample_ratio - 1.) * config.continue_step)
            if np.random.random() < ratio:
                if len(check_points) > traj_buf_size * start_sample_ratio:
                    max_step_remain = continue_step
                    idx, _ = sample_checkpoint_idx_by_rule(sample_rule)
                    if idx != -1:
                        cp = deepcopy(check_points[idx])
                        obs = restore_state(env, cp.state)
                        old_idx = idx
                        trajs = [(save_state(env), obs, 0, ep_len)]

        if ((t>int(9.5e5) and (t % config.eval_freq == 0)) or (
                t<=int(9.5e5) and t % (2*config.eval_freq) == 0)):
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
                    f"\n[#Step {t}] eval_reward: {eval_reward:.3f}, eval_step: {eval_step:.0f}, eval_time: {eval_time:.0f}, time: {log_info['time']:.3f}\n"
                    f"\tactor_loss: {log_info['actor_loss']:.3f}, critic_loss: {log_info['critic_loss']:.3f}, alpha_loss: {log_info['alpha_loss']:.3f}\n"
                    f"\tq1: {log_info['q1']:.3f}, target_q: {log_info['target_q']:.3f}, logp: {log_info['logp']:.3f}, alpha: {log_info['alpha']:.3f}\n"
                    f"\tbatch_reward: {batch.rewards.mean():.3f}, batch_reward_max: {batch.rewards.max():.3f}, batch_reward_min: {batch.rewards.min():.3f}\n"
                )
                logs.append(log_info)
            else:
                logs.append({"step": t, "reward": eval_reward})
                logger.info(
                    f"\n[#Step {t}] eval_reward: {eval_reward:.3f}, eval_time: {eval_time:.0f}\n"
                )

        # Save checkpoints
        if t % config.ckpt_freq == 0:
            agent.save(f"{ckpt_dir}", t // config.ckpt_freq)

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(
        f"{config.log_dir}/{config.env_name.lower()}/{exp_name}.csv")