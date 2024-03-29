import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "Walker2d-v2"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.lr = 3e-4
    config.seed = 2
    config.tau = 0.005
    config.gamma = 0.99
    config.batch_size = 256
    config.eval_episodes = 10
    config.start_timesteps = 25_000
    config.eval_freq = int(5e3)
    config.ckpt_freq = int(2e5)
    config.max_timesteps = int(1e6)
    config.expl_noise = 0.1
    config.hidden_dims = (256, 256)
    config.initializer = "glorot_uniform"
    return config