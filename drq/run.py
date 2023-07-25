import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import gym
import time
import random
import numpy as np

from tqdm import trange
from utils import get_logger, EfficientBuffer
from models import DrQLearner

# import jaxrl2.extra_envs.dm_control_suite
from dm_control import suite
from jaxrl2.wrappers import wrap_pixels


class Config:
    env_name = "cheetah-run-v0"
    seed = 0
    num_stack = 3
    image_size = 64

FLAGS = Config()


def make_env(env_name, seed):
    from wrappers import DMC2GYM
    domain_name, task_name = env_name.split('-')
    env = suite.load(domain_name=domain_name, task_name=task_name)
    env = DMC2GYM(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def eval_policy(agent, env, eval_episodes: int = 10):
    t1 = time.time()
    avg_reward, avg_step = 0., 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            avg_step += 1
            action = agent.sample_action(obs, eval_mode=True)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    avg_step /= eval_episodes
    return avg_reward, avg_step, time.time() - t1


def run():
    logger = get_logger(f"tmp.log")
    action_repeat = 4
    def wrap(env):
        if "quadruped" in FLAGS.env_name:
            camera_id = 2
        else:
            camera_id = 0
        return wrap_pixels(
            env,
            action_repeat=action_repeat,
            image_size=FLAGS.image_size,
            num_stack=FLAGS.num_stack,
            camera_id=camera_id,
        )

    np.random.seed(0)
    random.seed(0)

    env = make_env(FLAGS.env_name, 1)
    env = wrap(env)

    eval_env = make_env(FLAGS.env_name, 2)
    eval_env = wrap(eval_env)

    obs_shape = env.observation_space["pixels"].shape
    act_dim = env.action_space.shape[0]
    agent = DrQLearner(obs_shape, act_dim)

    buffer_size = int(5e5) // 4
    replay_buffer = EfficientBuffer(obs_shape, act_dim, max_size=buffer_size)
    replay_buffer_iterator = replay_buffer.get_iterator()

    observation, done = env.reset(), False

    for i in trange(1, 125001):
        if i < 1000:
            action = env.action_space.sample()
        else:
            action = agent.sample_action(observation)

        next_observation, reward, done, info = env.step(action)
        done_bool = int(done) if "TimeLimit.truncated" not in info else 0

        replay_buffer.add(observation["pixels"],
                          action,
                          next_observation["pixels"],
                          reward,
                          done_bool,
                          done)
        observation = next_observation

        if done:
            observation, done = env.reset(), False

        if i >= 1000:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)

        if (i * action_repeat) % 10000 == 0:
            eval_reward, eval_step, eval_time = eval_policy(agent, eval_env)
            logger.info(f"[{i*action_repeat}] R = {eval_reward:.2f}")


if __name__ == "__main__":
    run()
