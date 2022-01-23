import functools
import gym
import d4rl
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import DynamicsModel, GaussianMLP
from utils import ReplayBuffer, get_training_data
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"


def plot_val_loss():
    _, ax = plt.subplots()
    wds = [0.01, 0.05, 0.001, 0.005, 0.0001, 1e-5, 3e-5]
    for wd in wds:
        df = pd.read_csv(f'ensemble_models/hopper-medium-v2/s42_wd{wd}.csv', index_col=0)
        ax.plot(range(len(df)), df['val_loss'].values, label=f'{wd}')
        ax.legend()
    plt.savefig('val_losses.png')


def compute_td3_loss():
    seed = 42
    elite_num = 5
    ensemble_num = 7
    env_name = 'hopper-medium-v2'

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = DynamicsModel(env_name, seed, ensemble_num, elite_num)

    # load collected trajectories
    traj_data = np.load('saved_buffers/Hopper-v2/5agents.npz')
    observations = traj_data['observations']
    actions = traj_data['actions']
    next_observations = traj_data['next_observations']
    rewards = traj_data['rewards']
    dones = 1 - traj_data['discounts']

    inputs = np.concatenate([observations, actions], axis=-1)

    res = []
    Epochs = [20, 50, 150, 200]
    for epoch in tqdm(Epochs):
        model.load(f'saved_models/s2_b1024_e{epoch}')
        @jax.vmap
        def step(x):
            x = x.reshape(1, -1)
            model_mu, model_log_var = model.model.apply({"params": model.model_state.params}, x)  # (1, 12) ==> (7, 12)
            return model_mu

        ensemble_outputs = step(inputs)
        elite_idx = model.elite_mask.argmax(1)
        outputs = ensemble_outputs[:, elite_idx, :].mean(axis=1)

        pred_reward, pred_delta_obs = jnp.split(outputs, [1], axis=-1)  # (N, 1),  (N, 11)
        pred_next_obs = observations + pred_delta_obs
        reward_loss = jnp.mean(jnp.square(pred_reward - rewards))
        obs_loss = jnp.mean(jnp.mean(jnp.square(pred_next_obs - next_observations), axis=-1), axis=-1)

        res.append((epoch, reward_loss.item(), obs_loss.item()))


def check_model_valid_loss():
    env = gym.make('hopper-medium-v2')
    replay_buffer = ReplayBuffer(11, 3)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    inputs, targets, holdout_inputs, holdout_targets = get_training_data(replay_buffer)

    model = DynamicsModel('hopper-medium-v2', 2, 7, 5)
    model.load(f'ensemble_models/hopper-medium-v2/s42')

    @jax.jit
    def val_loss_fn(params, x, y):
        mu, log_var = jax.lax.stop_gradient(model.model.apply({"params": params}, x))  # (7, 14) ==> (7, 12), (7, 12)
        inv_var = jnp.exp(-log_var)  # (7, 12)
        mse_loss = jnp.mean(jnp.mean(jnp.square(mu - y), axis=-1), axis=-1)
        reward_loss = jnp.mean(jnp.square(mu[:, 0] - y[:, 0]), axis=-1)
        state_mse_loss = jnp.mean(jnp.mean(jnp.square(mu[:, 1:] - y[:, 1:]) * inv_var[:, 1:], axis=-1), axis=-1)
        state_var_loss = jnp.mean(jnp.mean(log_var[:, 1:], axis=-1), axis=-1)
        state_loss = state_mse_loss + state_var_loss
        return {"mse_loss": mse_loss, "reward_loss": reward_loss, "state_loss": state_loss}
    val_loss_fn = jax.vmap(val_loss_fn, in_axes=(None, 1, 1))

    losses = val_loss_fn(model.model_state.params, holdout_inputs, holdout_targets)
    losses = jax.tree_map(functools.partial(jnp.mean, axis=0), losses)
