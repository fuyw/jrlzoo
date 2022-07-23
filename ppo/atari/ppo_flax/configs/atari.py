import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Basic settings
    config.env_name = "PongNoFrameskip-v4"
    config.log_dir = "logs"
    config.batch_size = 256
    config.seed = 0

    # Total number of frames
    config.total_frames = int(1e7)
    config.log_num = 100

    # Parallel actor settings
    config.actor_num = 10
    config.rollout_len = 125

    # Training parameters
    config.lr = 2.5e-4
    config.gamma = 0.99

    # Number of training epochs per each unroll of the policy
    config.num_epochs = 3

    # GAE lambda
    config.lmbda = 0.95

    # PPO clip raio
    config.clip_param = 0.1

    # Value function loss weight
    config.vf_coeff = 0.5

    # Entropy loss weight
    config.entropy_coeff = 0.01

    # Linearly decay lr and clip parameter to zero
    config.decaying_lr_and_clip_param = False
    return config
