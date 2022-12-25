import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.algo = "dqn"
    config.env_name = "MountainCar-v0"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.lr = 1e-3
    config.seed = 0
    config.tau = 0.005
    config.gamma = 0.99
    config.batch_size = 256
    config.epsilon = 0.2
    config.eval_episodes = 10
    config.max_timesteps = int(25e4)
    config.start_timesteps = int(3e3)
    config.eval_freq = int(5e3)
    config.er = 'er'
    config.per_alpha = 0.6
    config.hidden_dims = (32, 32)
    return config
