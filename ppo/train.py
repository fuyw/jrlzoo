"""PPO Agent (~1000fps)
"""
from typing import List, Tuple
import os

from flax.training import checkpoints, train_state

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import collections
import ml_collections
import jax
import jax.numpy as jnp
import gym
import time
import optax
import numpy as np
import pandas as pd
from tqdm import trange
from flax  import linen as nn
from env_utils import get_num_actions
from utils import ReplayBuffer, get_logger
from agent import policy_action, RemoteSimulator
from models import ActorCritic


ExpTuple = collections.namedtuple(
    'ExpTuple', ['state', 'action', 'reward', 'value', 'log_prob', 'done'])


def eval_policy(agent,
                env: gym.Env,
                eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(agent.actor_state.params, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward, time.time() - t1


def get_experience(state: train_state.TrainState,
                   simulators: List[RemoteSimulator],
                   steps_per_actor: int):
    """Collect experience from agents.

    Runs `steps_per_actor` time steps of the game for each of the `simulators`.
    """
    all_experience = []
    for _ in range(steps_per_actor+1):
        sim_states = []
        for sim in simulators:
            sim_state = sim.conn.recv()
            sim_states.append(sim_state)
        sim_states = np.concatenate(sim_states, axis=0)
        log_probs, values = policy_action(state.apply_fn, state.params, sim_states)
        log_probs, values = jax.device_get((log_probs, values))
        probs = np.exp(np.array(log_probs))
        for i, sim in enumerate(simulators):
            probabilities = probs[i]
            action = np.random.choice(probs.shape[1], p=probabilities)
            sim.conn.send(action)
        experiences = []
        for i, sim in enumerate(simulators):
            sim_state, action, reward, done = sim.conn.recv()
            value = values[i, 0]
            log_prob = log_probs[i][action]
            sample = ExpTuple(sim_state, action, reward, value, log_prob, done)
            experiences.append(sample)
        all_experience.append(experiences)
    return all_experience


def create_train_state(params, model: nn.Module, configs: ml_collections.ConfigDict, train_steps: int) -> train_state.TrainState:
    if configs.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(init_value=configs.lr, end_value=0., trainsition_steps=train_steps)
    else:
        lr = configs.lr
    state = train_state.TranState.create(apply_fn=model.apply,
                                         params=params,
                                         tx=optax.adam(lr))
    return state


def get_initial_params(key: np.ndarray, model: nn.Module):
    input_dims = (1, 84, 84, 4)
    init_shape = jnp.ones(input_dims, jnp.float32)
    initial_params = model.init(key, init_shape)["params"]
    return initial_params


def train_and_evaluate(configs: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f'ppo_s{configs.seed}_{timestamp}'
    exp_info = f'# Running experiment for: {exp_name}_{configs.env_name} #'
    ckpt_dir = f"{configs.model_dir}/{configs.env_name.lower()}/{exp_name}"
    print('#' * len(exp_info) + f'\n{exp_info}\n' + '#' * len(exp_info))

    logger = get_logger(f'logs/{configs.env_name.lower()}/{exp_name}.log')
    logger.info(f"Exp configurations:\n{configs}")

    out_dim = get_num_actions(f"{configs.env_name}NoFrameskip-v4")

    # initialize the d4rl environment
    env = gym.make(configs.env_name)
    max_action = env.action_space.high[0]


    # remote agents
    simulators = [RemoteSimulator(f"{configs.env_name}NoFrameskip-v4")
                  for _ in range(configs.num_agents)]

    # training params
    loop_steps = configs.total_frames // (configs.num_agents * configs.actor_steps)

    # multiple steps
    iterations_per_step = (configs.num_agents * configs.actor_steps // configs.batch_size)

    # replay buffer
    # logs = [{
    #     "step": 0,
    #     "reward": eval_policy(agent, env, configs.eval_episodes)[0]
    # }]

    obs, done = env.reset(), False
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    for t in trange(1, configs.max_timesteps + 1):
        episode_timesteps += 1

    model = ActorCritic(out_dim)    
    initial_params = get_initial_params(jax.random.PRNGKey(0), model)
    state = create_train_state(initial_params, model, configs, loop_steps*configs.num_epochs*iterations_per_step)

    for step in range(loop_steps):
        alpha = 1. - step / loop_steps if configs.decaying_lr_and_clip_param else 1.
        all_experiences = get_experience(state, simulators, configs.actor_states)

    #     if t <= configs.start_timesteps:
    #         action = env.action_space.sample()
    #     else:
    #         action = (
    #             agent.sample_action(agent.actor_state.params, obs) +
    #             np.random.normal(
    #                 0, max_action * configs.expl_noise, size=act_dim)).clip(
    #                     -max_action, max_action)

    #     next_obs, reward, done, _ = env.step(action)
    #     done_bool = float(
    #         done) if episode_timesteps < env._max_episode_steps else 0

    #     replay_buffer.add(obs, action, next_obs, reward, done_bool)
    #     obs = next_obs
    #     episode_reward += reward

    #     if t > configs.start_timesteps:
    #         batch = replay_buffer.sample(configs.batch_size)
    #         log_info = agent.update(batch)

    #     if done:
    #         obs, done = env.reset(), False
    #         episode_reward = 0
    #         episode_timesteps = 0
    #         episode_num += 1

    #     if t % configs.eval_freq == 0:
    #         eval_reward, eval_time = eval_policy(agent, env,
    #                                              configs.eval_episodes)
    #         if t > configs.start_timesteps:
    #             log_info.update({
    #                 "step": t,
    #                 "reward": eval_reward,
    #                 "eval_time": eval_time,
    #                 "time": (time.time() - start_time) / 60
    #             })
    #             logger.info(
    #                 f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}, time: {log_info['time']:.2f}\n"
    #                 f"\tcritic_loss: {log_info['critic_loss']:.3f}, max_critic_loss: {log_info['max_critic_loss']:.3f}, min_critic_loss: {log_info['min_critic_loss']:.3f}\n"
    #                 f"\tactor_loss: {log_info['actor_loss']:.3f}, max_actor_loss: {log_info['max_actor_loss']:.3f}, min_actor_loss: {log_info['min_actor_loss']:.3f}\n"
    #                 f"\tq1: {log_info['q1']:.3f}, max_q1: {log_info['max_q1']:.3f}, min_q1: {log_info['min_q1']:.3f}\n"
    #                 f"\tq2: {log_info['q2']:.3f}, max_q2: {log_info['max_q2']:.3f}, min_q2: {log_info['min_q2']:.3f}\n"
    #                 f"\ttarget_q: {log_info['target_q']:.3f}, max_target_q: {log_info['max_target_q']:.3f}, min_target_q: {log_info['min_target_q']:.3f}\n"
    #             )
    #             logs.append(log_info)
    #         else:
    #             logs.append({"step": t, "reward": eval_reward})
    #             logger.info(
    #                 f"\n[#Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}\n"
    #             )

    #     # Save checkpoints
    #     if t % configs.ckpt_freq == 0:
    #         agent.save(f"{ckpt_dir}", t // configs.ckpt_freq)

    # log_df = pd.DataFrame(logs)
    # log_df.to_csv(
    #     f"{configs.log_dir}/{configs.env_name.lower()}/{exp_name}.csv")
