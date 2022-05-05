import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.2'

from typing import Tuple

import gym
import numpy as np
import pandas as pd
import time
import tqdm
from absl import app, flags
from ml_collections import config_flags

import sys
import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner
from utils import ReplayBuffer
import d4rl


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-medium-v2', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'configs/mujoco_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset, env_name):
    normalize_info_df = pd.read_csv('configs/minmax_traj_reward.csv', index_col=0).set_index('env_name')
    min_traj_reward, max_traj_reward = normalize_info_df.loc[env_name, ['min_traj_reward', 'max_traj_reward']]
    dataset.rewards = dataset.rewards / (max_traj_reward - min_traj_reward) * 1000


def make_env_and_dataset(env_name: str,
                         seed: int):
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def main(_):
    start_time = time.time()
    # env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    env = gym.make(FLAGS.env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # dataset = D4RLDataset(env)
    dataset = ReplayBuffer(obs_dim, act_dim)
    dataset.convert_D4RL(d4rl.qlearning_dataset(env))

    normalize(dataset, FLAGS.env_name)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    logs = [{'step': 0, 'reward':  evaluate(agent, env, FLAGS.eval_episodes)[0]['return']}]

    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.eval_interval == 0:
            eval_stats, eval_time = evaluate(agent, env, FLAGS.eval_episodes)
            update_info.update({
                'step': i,
                'reward': eval_stats['return'],
                'eval_time': eval_time,
                'time': (time.time() - start_time) / 60                
            })
            logs.append(update_info)
            print(f'#Step = {i}: reward={update_info["reward"]:.2f}, eval_time={update_info["eval_time"]:.2f}, actor_loss={update_info["actor_loss"]:.3f}, val_loss={update_info["value_loss"]:.1f}, q1={update_info["q1"]:.1f}')

    # Save logs
    os.makedirs('logs', exist_ok=True)
    os.makedirs(f"logs/{FLAGS.env_name}", exist_ok=True)
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"logs/{FLAGS.env_name}/{FLAGS.seed}.csv")


if __name__ == '__main__':
    app.run(main)
