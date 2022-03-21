import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "halfcheetah-medium-v2"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.initializer = "orthogonal"
    config.algo = "iql"
    config.hidden_dims = (256, 256)
    config.lr = 3e-4
    config.seed = 0
    config.tau = 0.005
    config.gamma = 0.99
    config.expectile = 0.7
    config.temperature = 3.0
    config.batch_size = 256
    config.log_freq = 10000
    config.eval_freq = 5000
    config.eval_episodes = 10
    config.max_timesteps = 1000000
    config.var_thresh = -5.0
    return config
