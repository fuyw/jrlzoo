import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "cheetah-run"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.lr = 3e-4
    config.seed = 0
    config.tau = 0.005
    config.gamma = 0.99
    config.emb_dim = 50
    config.batch_size = 256
    config.eval_episodes = 10
    config.start_timesteps = 1_000
    config.eval_freq = int(5e3)
    config.ckpt_freq = int(2e5)
    config.max_timesteps = int(1e6)
    config.hidden_dims = (256, 256)
    config.initializer = "orthogonal"

    config.image_size = 64
    config.num_stack = 3
    config.cnn_features = (32, 64, 128, 256)
    config.cnn_kernels = (3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = "VALID"

    return config