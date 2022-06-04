from typing import Callable
import gym
import time
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


###############
# Exp Setting #
###############
env_name = 'CartPole-v1'
num_cpu = 4  # Number of processes to use
n_timesteps = 25_000


###############################
# Multiprocessing RL Training #
###############################
def make_env(env_name: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    """
    def _init() -> gym.Env:
        env = gym.make(env_name)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def run_multi_process1():
    """
    Use vectorized environment (SubprocVecEnv), the actions sent
    to the wrapped env must be an array (one action per process).
    Also, observations, rewards and dones are arrays.
    """
    start_time = time.time()

    # Create the training environment
    env = SubprocVecEnv([make_env(env_name, i) for i in range(num_cpu)])

    # Initialize the multiprocess model
    model = A2C('MlpPolicy', env, verbose=0)

    # Create a separate environment for evaluation
    eval_env = gym.make(env_name)

    # Random agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Untrained agent -- Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')

    # Train the model
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time
    print(f'Took {total_time_multi:.2f}s for multiprocessed version1 - '
          f'{n_timesteps/total_time_multi:.2f} FPS')

    # Evaluate trained model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Trained agent -- Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')


# much faster than version1, better performance
def run_multi_process2():
    """
    The `make_vec_env()` helper does exactly the previous steps
    """
    start_time = time.time()

    # Create the training environment
    vec_env = make_vec_env(env_name, n_envs=num_cpu)

    # Initialize the multiprocess model
    vec_model = A2C('MlpPolicy', vec_env, verbose=0)

    # Create a separate environment for evaluation
    eval_env = gym.make(env_name)

    # Random agent, before training
    mean_reward, std_reward = evaluate_policy(vec_model, eval_env, n_eval_episodes=10)
    print(f'Untrained agent -- Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')

    # Train the model
    vec_model.learn(n_timesteps)
    total_time_multi = time.time() - start_time
    print(f'Took {total_time_multi:.2f}s for multiprocessed version2 - '
          f'{n_timesteps/total_time_multi:.2f} FPS')

    # Evaluate trained model
    mean_reward, std_reward = evaluate_policy(vec_model, eval_env, n_eval_episodes=10)
    print(f'Trained agent -- Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')


###########################
# Single Process Training #
###########################
def run_single_process():
    start_time = time.time()

    # Initialize the single process model
    single_process_model = A2C('MlpPolicy', env_name, verbose=0)

    # Create a separate environment for evaluation
    eval_env = gym.make(env_name)

    # Random agent, before training
    mean_reward, std_reward = evaluate_policy(single_process_model, eval_env, n_eval_episodes=10)
    print(f'Untrained agent -- Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')

    # Train the model
    single_process_model.learn(n_timesteps)
    total_time_multi = time.time() - start_time
    print(f'Took {total_time_multi:.2f}s for single process version - '
          f'{n_timesteps/total_time_multi:.2f} FPS')

    # Evaluate trained model
    mean_reward, std_reward = evaluate_policy(single_process_model, eval_env, n_eval_episodes=10)
    print(f'Trained agent -- Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')


if __name__ == '__main__':
    # run_multi_process1()
    run_multi_process2()
    print()
    run_single_process()

