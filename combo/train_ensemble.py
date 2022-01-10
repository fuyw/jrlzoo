"""
hopper-medium-v2 (256)
Epoch # 1: train_loss = -3.185 mse_loss = 0.949 var_loss = -4.135 val_loss = 0.752
Epoch # 2: train_loss = -5.297 mse_loss = 0.803 var_loss = -6.099 val_loss = 0.669
Epoch # 3: train_loss = -6.150 mse_loss = 0.696 var_loss = -6.846 val_loss = 0.600

hopper-medium-v2 (200)
Epoch # 1: train_loss = -3.225 mse_loss = 0.954 var_loss = -4.179 val_loss = 0.832
Epoch # 2: train_loss = -4.888 mse_loss = 0.882 var_loss = -5.770 val_loss = 0.683
Epoch # 3: train_loss = -5.749 mse_loss = 0.728 var_loss = -6.477 val_loss = 0.805
"""
from flax import linen as nn
from flax import serialization
from flax.core import FrozenDict
from flax.training import train_state
import functools
import gym
import d4rl
import jax
import jax.numpy as jnp
import numpy as np
import os
import optax
import pandas as pd
from tqdm import trange

from models import EnsembleDense, GaussianMLP
from utils import ReplayBuffer, get_training_data


def prepare_training_data(replay_buffer, holdout_ratio=0.1):
    # load the offline data
    observations = replay_buffer.observations
    actions = replay_buffer.actions
    next_observations = replay_buffer.next_observations
    rewards = replay_buffer.rewards.reshape(-1, 1)  # reshape for correct shape
    delta_observations = next_observations - observations

    # prepare for model inputs & outputs
    inputs = np.concatenate([observations, actions], axis=-1)
    targets = np.concatenate([rewards, delta_observations], axis=-1)

    # validation dataset
    num_holdout = int(inputs.shape[0] * holdout_ratio)
    permutation = np.random.permutation(inputs.shape[0])

    # split the dataset
    inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
    targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
    holdout_inputs = np.tile(holdout_inputs[None], [num_member, 1, 1])
    holdout_targets = np.tile(holdout_targets[None], [num_member, 1, 1])

    return inputs, targets, holdout_inputs, holdout_targets


def run(args):
    # Set experiment parameters
    lr = args.lr
    seed = args.seed
    epochs = args.epochs
    max_patience = args.max_patience
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_member = args.num_member
    holdout_ratio = args.holdout_ratio

    # Initialize the environment
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Set random seeds
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng, model_key = jax.random.split(rng, 2)

    # Initialize the probabilistic dynamics ensemble
    model = GaussianMLP(num_member, obs_dim+1)
    dummy_model_inputs = jnp.ones([num_member, obs_dim+act_dim], dtype=jnp.float32)
    model_params = model.init(model_key, dummy_model_inputs)["params"]
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model_params,
        tx=optax.adamw(learning_rate=lr, weight_decay=weight_decay))

    # Create the replay_buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    # Loss functions
    @jax.jit
    def loss_fn(params, x, y):
        mu, log_var = model.apply({'params': params}, x)
        inv_var = jnp.exp(-log_var)
        mse_loss = jnp.mean(jnp.mean(jnp.square(mu - y) * inv_var, axis=-1), axis=-1)
        var_loss = jnp.mean(jnp.mean(log_var, axis=-1), axis=-1)
        train_loss = mse_loss + var_loss
        return train_loss, {'mse_loss': mse_loss, 'var_loss': var_loss, 'train_loss': train_loss}

    @jax.jit
    def val_loss_fn(params, x, y):
        mu, log_var = jax.lax.stop_gradient(model.apply({'params': params}, x))
        inv_var = jnp.exp(-log_var)
        mse_loss = jnp.mean(jnp.mean(jnp.square(mu - y) * inv_var, axis=-1), axis=-1)
        return mse_loss

    # Wrap loss functions with jax.vmap
    grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 1, 1))
    val_loss_fn = jax.vmap(val_loss_fn, in_axes=(None, 1, 1))

    # Prepare for the train/val dataset
    inputs, targets, holdout_inputs, holdout_targets = get_training_data(
        replay_buffer, num_member, holdout_ratio)

    # Setting training parameters
    patience = 0
    batch_num = int(np.ceil(len(inputs) / batch_size))
    optimal_params = None
    min_val_loss = np.inf
    res = []

    # Train the model
    for epoch in trange(epochs):
        shuffled_idxs = np.concatenate([np.random.permutation(np.arange(
            inputs.shape[0])).reshape(1, -1) for _ in range(num_member)], axis=0)
        train_loss, mse_loss, var_loss = [], [], []
        for i in range(batch_num):
            batch_idxs = shuffled_idxs[:, i*batch_size:(i+1)*batch_size]  # (7, 256)
            batch_inputs = inputs[batch_idxs]       # (7, 256, 14)
            batch_targets = targets[batch_idxs]     # (7, 256, 12)

            (_, log_info), gradients = grad_fn(model_state.params, batch_inputs, batch_targets)
            log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
            gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
            model_state = model_state.apply_gradients(grads=gradients)

            train_loss.append(log_info['train_loss'].item())
            mse_loss.append(log_info['mse_loss'].item())
            var_loss.append(log_info['var_loss'].item())

        val_loss = jnp.mean(val_loss_fn(model_state.params, holdout_inputs, holdout_targets))
        if val_loss < min_val_loss:
            optimal_params = model_state.params
            min_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience == max_patience:
            print(f'Early stopping at epoch {epoch+1}.')
            break

        res.append((epoch, sum(train_loss)/batch_num, sum(mse_loss)/batch_num, sum(var_loss)/batch_num, val_loss))
        print(f'Epoch # {epoch+1}: train_loss = {sum(train_loss)/batch_num:.3f} '
              f'mse_loss = {sum(mse_loss)/batch_num:.3f} '
              f'var_loss = {sum(var_loss)/batch_num:.3f} '
              f'val_loss = {val_loss:.3f}')

    # Save logs
    res_df = pd.DataFrame(res, columns=['epoch', 'train_loss', 'mse_loss', 'var_loss', 'val_loss'])
    res_df.to_csv(f'{args.log_dir}/{args.env}/{seed}.csv')

    # Save the optimal model
    with open(f'{args.model_dir}/{args.env}/models_s{seed}.ckpt', 'wb') as f:
        f.write(serialization.to_bytes(optimal_params))

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hid_dim", default=200, type=int)
    parser.add_argument("--hid_layers", default=3, type=int)
    parser.add_argument("--num_member", default=7, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--weight_decay", default=3e-5, type=float)
    parser.add_argument("--batch_size", default=1280, type=int)
    parser.add_argument("--holdout_ratio", default=0.1, type=float)
    parser.add_argument("--max_patience", default=5, type=int)
    parser.add_argument("--model_dir", default="./ensemble_models", type=str)
    parser.add_argument("--log_dir", default="./logs", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f'{args.log_dir}/{args.env}', exist_ok=True)
    os.makedirs(f'{args.model_dir}/{args.env}', exist_ok=True)
    run(args)
