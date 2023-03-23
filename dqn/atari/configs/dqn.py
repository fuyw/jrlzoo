
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # General setting
    config.warmup_timesteps = int(5e4)
    config.total_timesteps = int(2.5e6)
    config.buffer_size = int(2.5e6)
    # config.total_timesteps = int(1e7)
    # config.buffer_size = int(1e6)
    config.update_target_freq = int(1e4)
    config.explore_frac = 0.1
    config.train_freq = 4
    config.batch_size = 32

    # Model parameters
    config.lr_start = 3e-4
    config.lr_end = 1e-5
    config.seed = 42
    config.tau = 0.005
    config.gamma = 0.99

    # Logging
    config.ckpt_num = 10
    config.eval_num = 50

    # Atari game
    config.env_name = "Breakout"
    config.contex_len = 4
    config.image_size = (84, 84)

    # Dirs
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.dataset_dir = "datasets"
    return config