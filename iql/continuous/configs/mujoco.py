import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "halfcheetah-medium-v2"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.initializer = "orthogonal"
    config.hidden_dims = (256, 256)
    config.lr = 3e-4
    config.seed = 0
    config.tau = 0.005
    config.gamma = 0.99
    config.expectile = 0.7
    config.adv_temperature = 3.0
    config.batch_size = 256
    config.eval_episodes = 10
    config.num_random = 10
    config.eval_freq = 5000
    config.max_timesteps = int(1e6)
    return config
