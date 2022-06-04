import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "Pong"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.seed = 0
    config.lr = 2.5e-4
    config.gamma = 0.99
    config.batch_size = 256
    config.start_timesteps = 25000
    config.total_frames = int(4e7)

    # parallel agents
    config.num_agents = 8

    # Number of steps each agent performs in one policy unroll. 
    config.actor_steps = 128

    # Number of training epochs per each unroll of the policy.
    config.num_epochs = 3

    # Generalized Advantage Estimation parameter.
    config.lmbda = 0.95

    # PPO clipping parameter
    config.clip_param = 0.1

    # Weight of value function loss in the total loss.
    config.vf_coeff = 0.5

    # Weight of entropy bonus in the total loss.
    config.entropy_coeff = 0.01

    # Linearly decay learning rate and clipping parameter to zero during training
    config.decaying_lr_and_clip_param = True

    config.eval_episodes = 10
    config.eval_freq = 100000
    config.ckpt_freq = 100000
    config.hidden_dims = (256, 256)
    config.initializer = "glorot_uniform"
    return config
