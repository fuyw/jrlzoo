import gym
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from constructor import construct_model
from fake_env import FakeEnv
from utils import ReplayBuffer


elite_id_dict = {
    'halfcheetah-medium': [5, 2, 0, 3, 6],
    'hopper-medium': [2, 6, 0, 3, 5],
    'walker2d-medium': [0, 5, 3, 1, 2],

    'halfcheetah-medium-expert': [6, 5, 1, 3, 0],
    'hopper-medium-expert': [2, 1, 3, 5, 4],
    'walker2d-medium-expert': [2, 6, 3, 1, 4],

    'hopper-medium-replay': [4, 3, 0, 2, 6],
    # 'halfcheetah-medium': [3, 4, 2, 5, 6],
    # 'halfcheetah-medium-replay': [6, 3, 2, 4, 5],
}

def load_model(env_name='halfcheetah-medium'):
    # Env parameters
    env = gym.make(f'{env_name}-v2')
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
        name=f'{env_name}_sn_smv_1_0',
        load_dir=f'tf_models/{env_name}',
        deterministic=False,
        multi_step_prediction=False,
        num_plan_steps=5,
        reward_classification=False
    )
    _model._model_inds = [5, 0, 3, 6, 2]
    fake_env = FakeEnv(
        _model,
        env_name.split('-')[0],
        penalty_coeff=0,
        penalty_learned_var=False,
        std_thresh=0,
        per_batch_std_percentile=0,
        oracle=False,
        oracle_env=env,
        model_rew_zero=False,
    )
    return fake_env
