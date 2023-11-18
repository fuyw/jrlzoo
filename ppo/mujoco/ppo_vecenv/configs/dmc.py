import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Basic settings
    config.env_name = "cheetah-run"
    config.model_dir = "saved_models"
    config.exp_name = "ppo"
    config.log_dir = "logs"
    config.batch_size = 256
    config.seed = 0

    # Total number of frames
    config.total_steps = int(1e7)
    config.log_num = 100

    # Envpool settings
    config.actor_num = 10
    config.rollout_len = 125

    # Training parameters
    config.lr = 3e-4
    config.gamma = 0.99
    config.hidden_dims = (64, 64)
    config.initializer = "orthogonal"

    # Number of training epochs per each unroll of the policy
    config.num_epochs = 4

    # GAE lambda
    config.lmbda = 0.95

    # PPO clip raio
    config.clip_param = 0.2

    # Value function loss weight
    config.vf_coeff = 0.5

    # Entropy loss weight
    config.entropy_coeff = 0.01

    # Target KL for early stopping
    config.target_kl = 0.01

    # Linearly decay lr and clip parameter to zero
    config.decaying_lr_and_clip_param = False
    return config
