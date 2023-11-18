import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "hopper-hop"
    config.exp_name = "a2c"
    config.seed = 0
    config.max_timesteps = int(1e6) * 10
    config.start_timesteps = 10000
    config.eval_episodes = 10

    config.eval_freq = config.max_timesteps // 100

    config.actor_num = 16
    config.rollout_len = 5

    return config