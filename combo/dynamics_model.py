"""
Epoch #1: train_loss = -5.224 mse_loss = 0.795 var_loss = -6.019 val_loss = 0.592

Epoch #1: train_loss = 3.622 mse_loss = 4.936 var_loss = -1.314 val_loss = 1.377
val_loss1 = [1.2827164 1.3246372 1.3936116 1.371917  1.3486533 1.3538406 1.5664071]

Epoch #2: train_loss = -1.313 mse_loss = 1.360 var_loss = -2.673 val_loss = 1.188
val_loss1 = [1.3106598 1.1248592 1.1366472 1.1385347 1.260187  1.1637954 1.178874 ]

Epoch #3: train_loss = -2.151 mse_loss = 1.230 var_loss = -3.381 val_loss = 1.063
val_loss1 = [1.2435447 1.0506101 1.0373491 1.1019484 1.0717213 0.9186476 1.0141898]

# Walker
Epoch #1: train_loss = -1.861 mse_loss = 1.713 var_loss = -3.574 val_loss = 1.318 val_loss1 = [1.2588706 1.1626441 1.344481  1.3021913 1.324056  1.543332  1.289835 ]

Epoch #2: train_loss = -3.320 mse_loss = 1.152 var_loss = -4.472 val_loss = 0.838 val_loss1 = [0.83917546 0.89436597 0.7926145  0.8648473  0.80759484 0.8354869
"""
from flax import linen as nn
from flax import serialization
from flax.training import train_state
import d4rl
import functools
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import pandas as pd
from tqdm import trange
from models import GaussianMLP
from utils import ReplayBuffer, get_training_data
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"


class DynamicsModel:
    def __init__(self,
                 env: str = "halfcheetah-medium-v2",
                 seed: int = 0,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 holdout_num: int = 1000,
                 lr: float = 1e-3,
                 weight_decay: float = 3e-5,
                 epochs: int = 100,
                 batch_size: int = 1280,
                 max_patience: int = 5,
                 model_dir: str = "./ensemble_models",
                 load_model: bool = False):

        # Model parameters
        self.seed = seed
        self.lr = lr
        self.weight_decay = weight_decay
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        self.elite_models = None
        self.holdout_num = holdout_num
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_patience = max_patience

        # Environment & ReplayBuffer
        self.env = gym.make(env)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim)
        self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env))

        # Initilaize the ensemble model
        np.random.seed(seed)
        rng = jax.random.PRNGKey(seed)
        _, model_key = jax.random.split(rng, 2)
        self.model = GaussianMLP(ensemble_num=ensemble_num, out_dim=obs_dim+1)
        dummy_model_inputs = jnp.ones([ensemble_num, obs_dim+act_dim], dtype=jnp.float32)
        model_params = self.model.init(model_key, dummy_model_inputs)["params"]
        self.model_state = train_state.TrainState.create(
            apply_fn=GaussianMLP.apply, params=model_params,
            tx=optax.adamw(learning_rate=lr, weight_decay=weight_decay))
        self.save_file = f"{model_dir}/{env}"

    def train(self):
        inputs, targets, holdout_inputs, holdout_targets = get_training_data(
            self.replay_buffer, self.ensemble_num, self.holdout_num)

        patience = 0 
        batch_num = int(np.ceil(len(inputs) / self.batch_size))
        min_val_loss = np.inf
        optial_params = None
        res = []

        # Loss functions
        @jax.jit
        def loss_fn(params, x, y):
            mu, log_var = self.model.apply({"params": params}, x)
            inv_var = jnp.exp(-log_var)
            mse_loss = jnp.mean(jnp.mean(jnp.square(mu - y) * inv_var, axis=-1), axis=-1)
            var_loss = jnp.mean(jnp.mean(log_var, axis=-1), axis=-1)
            train_loss = mse_loss + var_loss
            return train_loss, {"mse_loss": mse_loss, "var_loss": var_loss, "train_loss": train_loss}

        @jax.jit
        def val_loss_fn(params, x, y):
            mu, log_var = jax.lax.stop_gradient(self.model.apply({"params": params}, x))
            inv_var = jnp.exp(-log_var)
            mse_loss = jnp.mean(jnp.square(mu - y) * inv_var, axis=-1)
            return mse_loss

        # Wrap loss functions with jax.vmap
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 1, 1))
        val_loss_fn = jax.vmap(val_loss_fn, in_axes=(None, 1, 1))

        for epoch in trange(self.epochs):
            shuffled_idxs = np.concatenate([np.random.permutation(np.arange(
                inputs.shape[0])).reshape(1, -1) for _ in range(self.ensemble_num)], axis=0)
            train_loss, mse_loss, var_loss = [], [], []
            for i in range(batch_num):
                batch_idxs = shuffled_idxs[:, i*self.batch_size:(i+1)*self.batch_size]
                batch_inputs = inputs[batch_idxs]  # (7, 256, 14)
                batch_targets = targets[batch_idxs]  # (7, 256, 12)

                (_, log_info), gradients = grad_fn(self.model_state.params, batch_inputs, batch_targets)
                log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
                gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
                self.model_state = self.model_state.apply_gradients(grads=gradients)

                train_loss.append(log_info["train_loss"].item())
                mse_loss.append(log_info["mse_loss"].item())
                var_loss.append(log_info["var_loss"].item())
        
            val_loss = jnp.mean(val_loss_fn(self.model_state.params, holdout_inputs, holdout_targets), axis=0)  # (7,)
            mean_val_loss = jnp.mean(val_loss)
            if mean_val_loss < min_val_loss:
                optimal_params = self.model_state.params
                min_val_loss = mean_val_loss
                elite_models = jnp.argsort(val_loss)  # find elite models
                patience = 0
            else:
                patience += 1
            if patience == self.max_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

            res.append((epoch, sum(train_loss)/batch_num, sum(mse_loss)/batch_num, sum(var_loss)/batch_num, mean_val_loss))
            print(f"Epoch #{epoch+1}: train_loss = {sum(train_loss)/batch_num:.3f} "
                  f"mse_loss = {sum(mse_loss)/batch_num:.3f} "
                  f"var_loss = {sum(var_loss)/batch_num:.3f} "
                  f"val_loss = {mean_val_loss:.3f} "
                  f"val_loss1 = {val_loss}")

        res_df = pd.DataFrame(res, columns=["epoch", "train_loss", "mse_loss", "var_loss", "val_loss"])
        res_df.to_csv(f"{self.save_file}/s{self.seed}.csv")
        with open(f"{self.save_file}/s{self.seed}.ckpt", "wb") as f:
            f.write(serialization.to_bytes(optimal_params))
        with open(f"{self.save_file}/s{self.seed}_elite_models.txt", "w") as f:
            for idx in elite_models:
                f.write(f"{idx}\n")

    def load(self, filename):
        with open(f"{filename}.ckpt", "rb") as f:
            model_params = serialization.from_bytes(
                self.model_state.params, f.read())
        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=model_params,
            tx=optax.adamw(learning_rate=self.lr,
            weight_decay=self.weight_decay))

    def test_load_model(self, filename):
        self.load(filename)
        inputs, targets, holdout_inputs, holdout_targets = get_training_data(
            self.replay_buffer, self.ensemble_num, self.holdout_num) 

        @jax.jit
        def val_loss_fn(params, x, y):
            mu, log_var = jax.lax.stop_gradient(self.model.apply({"params": params}, x))
            inv_var = jnp.exp(-log_var)
            mse_loss = jnp.mean(jnp.square(mu - y) * inv_var, axis=-1)
            return mse_loss

        # Wrap loss functions with jax.vmap
        val_loss_fn = jax.vmap(val_loss_fn, in_axes=(None, 1, 1))
        val_loss = jnp.mean(val_loss_fn(self.model_state.params, holdout_inputs, holdout_targets), axis=0)
        mean_val_loss = jnp.mean(val_loss)
        print(val_loss)
        print(mean_val_loss)


if __name__ == "__main__":
    env = "walker2d-medium-v2"
    dynamics_model = DynamicsModel(env)
    dynamics_model.train()
    # dynamics_model.test_load_model("ensemble_models/hopper-medium-v2/s0")
