import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.2'

from typing import Tuple

import gym
import d4rl
import numpy as np
import pandas as pd
import time
import tqdm
from utils import ReplayBuffer
from absl import app, flags
from ml_collections import config_flags

import sys
import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner


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



def normalize(dataset, env_name, eps=1e-5):
    normalize_info_df = pd.read_csv('configs2/minmax_traj_reward.csv', index_col=0).set_index('env_name')
    min_traj_reward, max_traj_reward = normalize_info_df.loc[env_name, ['min_traj_reward', 'max_traj_reward']]
    dataset.rewards = dataset.rewards / (max_traj_reward - min_traj_reward) * 1000

def normalize_buffer(buffer, env_name, eps=1e-5):
    normalize_info_df = pd.read_csv('configs2/minmax_traj_reward.csv', index_col=0).set_index('env_name')
    min_traj_reward, max_traj_reward = normalize_info_df.loc[env_name, ['min_traj_reward', 'max_traj_reward']]
    buffer.rewards = buffer.rewards / (max_traj_reward - min_traj_reward) * 1000
    lim = 1 - eps
    buffer.actions = np.clip(buffer.actions, -lim, lim)


def main(_):
    FLAGS(sys.argv)
    start_time = time.time()
    env = gym.make(FLAGS.env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # dataset = D4RLDataset(env)
    # normalize(dataset, FLAGS.env_name)

    dataset = ReplayBuffer(obs_dim, act_dim)
    dataset.convert_D4RL(d4rl.qlearning_dataset(env))
    normalize_buffer(dataset, FLAGS.env_name)

    # abs(dataset.masks - buffer.discounts.squeeze()).reshape(-1).max()
    # normalize2(dataset, FLAGS.env_name)

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
            print(f'#Step = {i}: reward={update_info["reward"]:.2f}, actor_loss={update_info["actor_loss"]:.3f}, val_loss={update_info["value_loss"]:.1f}, logp={update_info["log_probs"]:.1f}, q1={update_info["q1"]:.1f}')

    # Save logs
    os.makedirs('logs', exist_ok=True)
    os.makedirs(f"logs/{FLAGS.env_name}", exist_ok=True)
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"logs/{FLAGS.env_name}/{FLAGS.seed}.csv")


if __name__ == '__main__':
    app.run(main)
