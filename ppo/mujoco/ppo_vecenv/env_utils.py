from typing import List

import gym


def create_vec_env(env_name: str, num_envs: int, seeds: List[int] = None, sync: bool=True):
    def create_env():
        env = gym.make(env_name)
        return env
    env_fns = [create_env] * num_envs
    if sync:
        envs = gym.vector.SyncVectorEnv(env_fns)
    else:
        envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True)
    envs.seed(seeds)
    return envs