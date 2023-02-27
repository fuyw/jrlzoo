import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()

    # General settings
    config.env_name = "antmaze-medium-play-v0"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.initializer = "orthogonal"

    # Train and eval
    config.eval_episodes = 100
    config.max_timesteps = int(2.5e5)
    # config.max_timesteps = int(1e6)
    config.eval_freq = config.max_timesteps // 10
    config.start_timesteps = 2500
    config.update_num = 10

    # Hyperparameters
    config.seed = 0
    config.lr = 3e-4
    config.hidden_dims = (256, 256)
    config.batch_size = 256

    # GDE setting
    config.algo = "iql"
    config.base_algo = "iql"
    config.nstep = 3
    config.lmbda = 0.3
    config.buffer_size = int(2.5e5)

    # IQL settings
    config.tau = 0.005
    config.gamma = 0.99
    config.expectile = 0.9
    config.adv_temperature = 10.0

    return config
