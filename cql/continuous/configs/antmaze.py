import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "antmaze-umaze-v0"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.initializer = "orthogonal"
    config.actor_hidden_dims = (256, 256)
    config.critic_hidden_dims = (256, 256, 256)
    config.lr_critic = 3e-4
    config.lr_actor = 1e-4
    config.seed = 0
    config.tau = 0.005
    config.gamma = 0.99
    config.batch_size = 256
    config.num_random = 10
    config.min_q_weight = 5.0
    config.target_entropy = None
    config.backup_entropy = False

    config.max_target_backup = True
    config.with_lagrange = True
    config.lagrange_thresh = 0.2
    config.cql_clip_diff_min = -200 
    config.cql_clip_diff_max = np.inf
    config.ckpt_freq = 100_000
    config.eval_freq = 100_000
    config.eval_episodes = 100
    config.bc_timesteps = 40_000
    config.max_timesteps = 1_000_000

    return config
