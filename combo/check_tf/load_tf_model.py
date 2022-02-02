"""
def report_progress():
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv('/usr/local/data/yuweifu/jaxrl/benchmarks/combo/ray_combo/hopper/hopper_medium_nolip_len5_minq_3.0_lagrange_0.0_deterministic_backup_temp1.0_lr3e-5_3layers_normalize_for_fake_states_rollout_random_real0.5_iter2000_2000e3/seed:55_2022-01-31_23-19-41tvfrb311/progress.csv')
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
    for idx, col in enumerate(['evaluation/return-average', 'Q-avg', 'Q_loss', 'min_Q_loss']):
        ax = axes[idx]
        ax.plot(df['timesteps_total'].values, df[col].values)
        ax.set_title(col)
    plt.savefig('check_tf_combo.png')
"""
import d4rl
import gym
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from constructor import construct_model
from fake_env import FakeEnv
from utils import ReplayBuffer


def load_model():
    # Env parameters
    env = gym.make('hopper-medium-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    _model = construct_model(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=200,
        num_layers=4,
        model_lr=1e-3,
        num_networks=7,
        num_elites=5,
        model_type='mlp',
        sn=False,
        gradient_penalty=0.0,
        gradient_penalty_scale=10.0,
        separate_mean_var=False,
        no_sn_last=False,
        name='hopper-medium_sn_smv_1_0',
        load_dir='/usr/local/data/yuweifu/jaxrl/benchmarks/combo/ray_combo/saved_models',
        deterministic=False,
        multi_step_prediction=False,
        num_plan_steps=5,
        reward_classification=False
    )
    _model._model_inds = [1, 4, 6, 0, 5]
    fake_env = FakeEnv(
        _model,
        'hopper',
        penalty_coeff=0,
        penalty_learned_var=False,
        std_thresh=0,
        per_batch_std_percentile=0,
        oracle=False,
        oracle_env=env,
        model_rew_zero=False,
    )
    return fake_env
