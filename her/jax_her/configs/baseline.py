from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.env_name = "FetchPickAndPlace-v3"
    config.model = "ddpg"
    config.seed = 42
    config.num_envs = 16
    config.max_timesteps = int(2e6)
    config.start_timesteps = 3000
    config.eval_episodes = 50
    config.hidden_dims = (256, 256)
    config.max_episode_steps = 100
    config.eval_freq = config.max_timesteps // 50
    config.ckpt_freq = config.max_timesteps // 10
    config.save_video = False

    config.lr = 3e-4
    config.gamma = 0.98

    # ddpg exploration noise
    config.expl_noise = 0.1

    # HER
    config.replay_k = 4
    config.batch_size = 256
    config.max_size = int(1e6)

    return config

