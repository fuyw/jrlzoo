import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()

    # General settings
    config.env_name = "halfcheetah-medium-v2"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.initializer = "orthogonal"

    # Train and eval
    config.eval_episodes = 10
    config.max_timesteps = int(2.5e5)
    config.eval_freq = config.max_timesteps // 100
    config.start_timesteps = 2500

    # Hyperparameters
    config.seed = 0
    config.lr = 3e-4
    config.hidden_dims = (256, 256)
    config.batch_size = 256

    # GDE setting
    config.algo = "iql"
    config.base_algo = "iql"

    # IQL settings
    config.tau = 0.005
    config.gamma = 0.99
    config.expectile = 0.7
    config.adv_temperature = 3.0

    return config
