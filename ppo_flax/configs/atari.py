import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Atari game name
    config.env_name = "PongNoFrameskip-v4"

    # Total number of frames
    config.total_frames = int(1e7)

    config.log_dir = "logs"
    config.lr = 3e-4
    config.seed = 0
    config.batch_size = 256

    # Number of training epochs per each unroll of the policy
    config.num_epochs = 3

    # Number of agents playing in parallel
    config.num_agents = 4

    # Number of steps each agent performs in one policy rollout
    config.actor_steps = 128

    config.gamma = 0.99
    config.lmbda = 0.95
    config.clip_param = 0.1

    # Value function loss weight
    config.vf_coeff = 0.5

    # Entropy loss weight
    config.entropy_coeff = 0.01

    # Linearly decay lr and clip parameter to zero
    config.decaying_lr_and_clip_param = True
    return config

