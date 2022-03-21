import functools
from typing import Any, Callable

from flax import linen as nn
from flax import serialization
from flax.training import train_state, checkpoints
import d4rl
import gym
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
from tqdm import trange

from static_fn import static_fns
from utils import ReplayBuffer, get_training_data


class EnsembleDense(nn.Module):
    ensemble_num: int
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable = jax.nn.initializers.glorot_uniform()
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.array) -> jnp.array:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param("kernel", self.kernel_init,
                            (self.ensemble_num, inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.einsum("ij,ijk->ik", inputs, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.ensemble_num, self.features))
            bias = jnp.asarray(bias, self.dtype)
            y += bias
        return y


class GaussianMLP(nn.Module):
    ensemble_num: int
    out_dim: int
    hid_dim: int = 200
    max_log_var: float = 0.5
    min_log_var: float = -10.0

    def setup(self):
        self.l1 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.hid_dim, name="fc1")
        self.l2 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.hid_dim, name="fc2")
        self.l3 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.hid_dim, name="fc3")
        self.mean_and_logvar = EnsembleDense(ensemble_num=self.ensemble_num, features=self.out_dim*2, name="output")

    def __call__(self, x):
        x = nn.leaky_relu(self.l1(x))
        x = nn.leaky_relu(self.l2(x))
        x = nn.leaky_relu(self.l3(x))
        x = self.mean_and_logvar(x)

        mu, log_var = jnp.split(x, 2, axis=-1)
        log_var = self.max_log_var - jax.nn.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + jax.nn.softplus(log_var - self.min_log_var)
        return mu, log_var


class DynamicsModel:
    def __init__(self,
                 env_name: str = "halfcheetah-medium-v2",
                 seed: int = 42,
                 holdout_ratio: float = 0.05,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 epochs: int = 100,
                 batch_size: int = 256,
                 max_patience: int = 10,
                 dynamics_model_dir: str = "./saved_dynamics_models"):

        # Model parameters
        self.env_name = env_name
        self.seed = seed
        self.lr = lr
        self.static_fn = static_fns[env_name.split('-')[0].lower()]
        print(f'Load static_fn: {self.static_fn}')
        self.weight_decay = weight_decay
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        self.elite_models = None
        self.holdout_ratio = holdout_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_patience = max_patience

        # Environment & ReplayBuffer
        print(f'Loading data for {env_name}')
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim)
        self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env))

        # Initilaize saving settings
        self.save_dir = f"{dynamics_model_dir}/{env_name}"
        self.elite_mask = np.eye(self.ensemble_num)[range(elite_num), :]

        # Initilaize the ensemble model
        rng = jax.random.PRNGKey(seed)
        _, model_key = jax.random.split(rng, 2)
        self.model = GaussianMLP(ensemble_num=ensemble_num, out_dim=self.obs_dim+1)
        dummy_model_inputs = jnp.ones([ensemble_num, self.obs_dim+self.act_dim], dtype=jnp.float32)
        model_params = self.model.init(model_key, dummy_model_inputs)["params"]

        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=model_params,
            tx=optax.adamw(learning_rate=self.lr, weight_decay=self.weight_decay))

        # Normalize inputs
        self.obs_mean = 0
        self.obs_std = 1

    def load(self):
        filename = f"old_saved_dynamics_models/{self.env_name}"
        with open(f"{filename}/dynamics_model.ckpt", "rb") as f:
            model_params = serialization.from_bytes(
                self.model_state.params, f.read())
        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=model_params,
            tx=optax.adamw(learning_rate=self.lr,
            weight_decay=self.weight_decay))
        elite_idx = np.loadtxt(f'{filename}/elite_models.txt', dtype=np.int32)[:self.elite_num]
        self.elite_mask = np.eye(self.ensemble_num)[elite_idx, :]
        normalize_stat = np.load(f'{filename}/normalize_stat.npz')
        self.obs_mean = normalize_stat['obs_mean'].squeeze()
        self.obs_std = normalize_stat['obs_std'].squeeze()

    def load_new(self):
        self.model_state = checkpoints.restore_checkpoint(ckpt_dir=f"{self.save_dir}",
                                                          target=self.model_state,
                                                          step=0,
                                                          prefix="dynamics_model_")
        elite_idx = np.loadtxt(f"{self.save_dir}/elite_models.txt", dtype=np.int32)[:self.elite_num]
        self.elite_mask = np.eye(self.ensemble_num)[elite_idx, :]
        normalize_stat = np.load(f'{self.save_dir}/normalize_stat.npz')
        self.obs_mean = normalize_stat['obs_mean']
        self.obs_std = normalize_stat['obs_std']

    def train(self):
        (inputs, targets, holdout_inputs, holdout_targets, self.obs_mean, self.obs_std) = get_training_data(
            self.replay_buffer, self.ensemble_num, self.holdout_ratio)
        patience = 0 
        batch_num = int(np.ceil(len(inputs) / self.batch_size))
        min_val_loss = np.inf
        res = []
        print(f'batch_num     = {batch_num}')
        print(f'inputs.shape  = {inputs.shape}')
        print(f'targets.shape = {targets.shape}') 
        print(f'holdout_inputs.shape  = {holdout_inputs.shape}')
        print(f'holdout_targets.shape = {holdout_targets.shape}') 

        # Loss functions
        @jax.jit
        def train_step(model_state: train_state.TrainState, batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray):
            def loss_fn(params, x, y):
                mu, log_var = self.model.apply({"params": params}, x)  # (7, 14) ==> (7, 12)
                inv_var = jnp.exp(-1.0 * log_var)    # (7, 12)
                mse_loss = jnp.square(mu - y)  # (7, 12)
                train_loss = jnp.mean(mse_loss * inv_var + log_var, axis=-1).sum()
                return train_loss, {"mse_loss": mse_loss.mean(),
                                    "var_loss": log_var.mean(),
                                    "train_loss": train_loss}
            grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 1, 1))
            (_, log_info), gradients = grad_fn(model_state.params, batch_inputs, batch_targets)
            log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
            gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
            new_model_state = model_state.apply_gradients(grads=gradients)
            return new_model_state, log_info

        @jax.jit
        def eval_step(model_state: train_state.TrainState, batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray):
            def loss_fn(params, x, y):
                mu, _ = jax.lax.stop_gradient(self.model.apply({"params": params}, x))  # (7, 14) ==> (7, 12)
                mse_loss = jnp.mean(jnp.square(mu - y), axis=-1)  # (7, 12) ==> (7,)
                reward_loss = jnp.square(mu[:, -1] - y[:, -1]).mean()
                state_loss = jnp.square(mu[:, :-1] - y[:, :-1]).mean()
                return mse_loss, {"reward_loss": reward_loss, "state_loss": state_loss}
            loss_fn = jax.vmap(loss_fn, in_axes=(None, 1, 1))
            loss, log_info = loss_fn(model_state.params, batch_inputs, batch_targets)
            log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
            return loss, log_info

        for epoch in trange(self.epochs):
            shuffled_idxs = np.concatenate([np.random.permutation(np.arange(
                inputs.shape[0])).reshape(1, -1) for _ in range(self.ensemble_num)], axis=0)  # (7, 1000000)
            train_loss, mse_loss, var_loss = [], [], []
            for i in range(batch_num):
                batch_idxs = shuffled_idxs[:, i*self.batch_size:(i+1)*self.batch_size]
                batch_inputs = inputs[batch_idxs]    # (7, 256, 14)
                batch_targets = targets[batch_idxs]  # (7, 256, 12)

                self.model_state, log_info = train_step(self.model_state, batch_inputs, batch_targets)
                train_loss.append(log_info["train_loss"].item())
                mse_loss.append(log_info["mse_loss"].item())
                var_loss.append(log_info["var_loss"].item())

            val_loss, val_info = eval_step(self.model_state, holdout_inputs, holdout_targets)
            val_loss = jnp.mean(val_loss, axis=0)  # (N, 7) ==> (7,)
            mean_val_loss = jnp.mean(val_loss)
            if mean_val_loss < min_val_loss:
                optimal_state = self.model_state
                min_val_loss = mean_val_loss
                elite_models = jnp.argsort(val_loss)  # find elite models
                patience = 0
            else:
                patience += 1
            if epoch > 20 and patience > self.max_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

            res.append((epoch, sum(train_loss)/batch_num, sum(mse_loss)/batch_num, sum(var_loss)/batch_num, mean_val_loss))
            print(f"Epoch #{epoch+1}: "
                  f"train_loss={sum(train_loss)/batch_num:.3f}\t"
                  f"mse_loss={sum(mse_loss)/batch_num:.3f}\t"
                  f"var_loss={sum(var_loss)/batch_num:.3f}\t"
                  f"val_loss={mean_val_loss:.3f}\t"
                  f"val_rew_loss={val_info['reward_loss']:.3f}\t"
                  f"val_state_loss={val_info['state_loss']:.3f}")

        checkpoints.save_checkpoint(f"{self.save_dir}", optimal_state, 0, prefix="dynamics_model_", overwrite=True)
        res_df = pd.DataFrame(res, columns=["epoch", "train_loss", "mse_loss", "var_loss", "val_loss"])
        res_df.to_csv(f"{self.save_dir}/train_log.csv")
        ckpt_loss, ckpt_info = eval_step(optimal_state, holdout_inputs, holdout_targets)
        ckpt_loss = jnp.mean(ckpt_loss, axis=0)
        with open(f"{self.save_dir}/elite_models.txt", "w") as f:
            for idx in elite_models:
                f.write(f"{idx}\n")
        elite_idx = elite_models.to_py()[:self.elite_num]
        self.elite_mask = np.eye(self.ensemble_num)[elite_idx, :]
        np.savez(f"{self.save_dir}/normalize_stat", obs_mean=self.obs_mean, obs_std=self.obs_std)

    def normalize(self, observations):
        new_observations = (observations - self.obs_mean) / self.obs_std
        return new_observations

    def denormalize(self, observations):
        new_observations = observations * self.obs_std + self.obs_mean
        return new_observations
    
    @functools.partial(jax.jit, static_argnames=("self"))
    def rollout2(self, key, params, observations, actions, model_masks):
        def rollout_fn(observation, action, model_mask):
            x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
            model_mu, model_log_var = self.model.apply({"params": params}, x)
            observation_mu, reward_mu = jnp.split(model_mu, [self.obs_dim], axis=-1)

            model_std = jnp.sqrt(jnp.exp(model_log_var))
            model_noise = model_std * jax.random.normal(key, (self.ensemble_num, self.obs_dim+1))
            observation_noise, reward_noise = jnp.split(model_noise, [self.obs_dim], axis=-1)

            model_next_observation = observation + jnp.sum(model_mask * observation_mu + observation_noise, axis=0)
            model_reward = jnp.sum(model_mask * reward_mu, axis=0)
            return model_next_observation, model_reward
        next_observations, rewards = jax.vmap(rollout_fn, in_axes=(0, 0, 0))(observations, actions, model_masks)
        next_observations = self.denormalize(next_observations)
        return next_observations, rewards

    # @functools.partial(jax.jit, static_argnames=("self"))
    def rollout(self, params, observations, actions, model_masks):
        @jax.jit
        def rollout_fn(observation, action, model_mask):
            x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
            model_mu, model_log_var = self.model.apply({"params": params}, x)
            max_log_var = model_log_var.mean(-1).max() 
            observation_mu, reward_mu = jnp.split(model_mu, [self.obs_dim], axis=-1)
            model_next_observation = observation + jnp.sum(model_mask * observation_mu, axis=0)
            model_reward = jnp.sum(model_mask * reward_mu, axis=0)
            return model_next_observation, model_reward, max_log_var
        next_observations, rewards, max_log_vars = jax.vmap(rollout_fn, in_axes=(0, 0, 0))(observations, actions, model_masks)
        next_observations = self.denormalize(next_observations)
        return next_observations, rewards, max_log_vars

    def step(self, key, observations, actions):
        model_idx = jax.random.randint(key, shape=(actions.shape[0],), minval=0, maxval=self.elite_num)
        model_masks = self.elite_mask[model_idx].reshape(-1, self.ensemble_num, 1)
        next_observations, rewards, max_log_vars = self.rollout(self.model_state.params, observations, actions, model_masks)
        terminals = self.static_fn.termination_fn(observations, actions, next_observations)
        return next_observations, rewards.squeeze(), terminals.squeeze(), max_log_vars
