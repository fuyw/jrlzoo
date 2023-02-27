import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # General settings
    config.env_name = "walker2d-random-v2"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.initializer = "orthogonal"

    # Train and eval
    config.eval_episodes = 10
    config.start_timesteps = 2500
    config.max_timesteps = int(1e6) // 4
    config.eval_freq = config.max_timesteps//100

    # Hyperparameters
    config.seed = 0
    config.lr = 3e-4
    config.hidden_dims = (256, 256)
    config.batch_size = 256

    # Algo
    config.algo = "iql"
    config.finetune = "naive"

    # Finetune
    config.offline_buffer = False

    # IQL settings
    config.tau = 0.005
    config.gamma = 0.99
    config.expectile = 0.7
    config.adv_temperature = 3.0
    config.std_temperature = 1.0

    return config
