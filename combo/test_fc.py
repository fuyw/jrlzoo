from typing import Any, Callable, Optional
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
import functools
import gym
import d4rl
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from utils import ReplayBuffer


class EnsembleDense(nn.Module):
    num_members: int
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.array) -> jnp.array:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param("kernel", self.kernel_init,
                            (self.num_members, inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.einsum("ij,ijk->ik", inputs, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.num_members, self.features))
            bias = jnp.asarray(bias, self.dtype)
            y += bias
        return y


class GaussianMLP(nn.Module):
    num_members: int
    out_dim: int
    hid_dim: int = 256
    max_log_var: float = 0.5
    min_log_var: float = -10.0

    def setup(self):
        self.l1 = EnsembleDense(num_members=self.num_members, features=self.hid_dim, name="fc1")
        self.l2 = EnsembleDense(num_members=self.num_members, features=self.hid_dim, name="fc2")
        self.l3 = EnsembleDense(num_members=self.num_members, features=self.hid_dim, name="fc3")
        self.l4 = EnsembleDense(num_members=self.num_members, features=self.hid_dim, name="fc4")
        self.l5 = EnsembleDense(num_members=self.num_members, features=self.out_dim*2, name="fc5")

    def __call__(self, x):
        x = nn.swish(self.l1(x))
        x = nn.swish(self.l2(x))
        x = nn.swish(self.l3(x))
        x = nn.swish(self.l4(x))
        x = self.l5(x)

        mu, log_var = jnp.split(x, 2, axis=-1)
        # TODO:
        # log_var = jnp.clip(log_std, self.min_log_var, self.max_log_var)
        log_var = self.max_log_var - jax.nn.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + jax.nn.softplus(log_var - self.min_log_var)
        return mu, log_var


env = gym.make('hopper-medium-v2')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

lr_model = 1e-3
weight_decay = 3e-5
num_members = 7
holdout_ratio = 0.1


replay_buffer = ReplayBuffer(obs_dim, act_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))


env.seed(0)
env.action_space.seed(0)
np.random.seed(0)

rng = jax.random.PRNGKey(0)
rng, model_key = jax.random.split(rng, 2)

dummy_model_inputs = jnp.ones([num_members, obs_dim+act_dim], dtype=jnp.float32)
model = GaussianMLP(num_members, obs_dim+1)
model_params = model.init(model_key, dummy_model_inputs)['params']
model_state = train_state.TrainState.create(
    apply_fn=GaussianMLP.apply,
    params=model_params,
    tx=optax.adamw(learning_rate=lr_model, weight_decay=weight_decay)
)


observations = replay_buffer.observations
actions = replay_buffer.actions
next_observations = replay_buffer.next_observations
rewards = replay_buffer.rewards.reshape(-1, 1)
delta_observations = next_observations - observations

inputs = np.concatenate([observations, actions], axis=-1)
targets = np.concatenate([rewards, delta_observations], axis=-1)

num_holdout = int(inputs.shape[0] * holdout_ratio)
permutation = np.random.permutation(inputs.shape[0])

inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
holdout_inputs = np.tile(holdout_inputs[None], [num_members, 1, 1])
holdout_targets = np.tile(holdout_targets[None], [num_members, 1, 1])

batch_size = 256
batch_num = int(np.ceil(inputs.shape[0] / batch_size))


@jax.jit
def loss_fn(params, x, y):
    mu, log_var = model.apply({'params': params}, x)  # (7, 12)
    inv_var = jnp.exp(-log_var)
    mse_loss = jnp.mean(jnp.mean(jnp.square(mu - y) * inv_var, axis=-1), axis=-1)
    var_loss = jnp.mean(jnp.mean(log_var, axis=-1), axis=-1)
    total_loss = mse_loss + var_loss
    return total_loss, {'mse_loss': mse_loss, 'var_loss': var_loss, 'model_loss': total_loss}

grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 1, 1))


for epoch in trange(50):
    shuffled_idxs = np.concatenate([np.random.permutation(np.arange(inputs.shape[0])).reshape(1, -1)
                                    for _ in range(num_members)], axis=0)
    model_loss, mse_loss, var_loss = [], [], []
    for i in range(batch_num):
        batch_idxs = shuffled_idxs[:, i*batch_size:(i+1)*batch_size]  # (7, 256)
        batch_inputs = inputs[batch_idxs]       # (7, 256, 14)
        batch_targets = targets[batch_idxs]     # (7, 256, 12)

        (_, log_info), gradients = grad_fn(model_state.params, batch_inputs, batch_targets)
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        model_state = model_state.apply_gradients(grads=gradients)

        model_loss.append(log_info['model_loss'].item())
        mse_loss.append(log_info['mse_loss'].item())
        var_loss.append(log_info['var_loss'].item())
    print(f'Epoch # {epoch+1}: model_loss = {sum(model_loss)/batch_num:.2f} '
          f'mse_loss = {sum(mse_loss)/batch_num:.2f} '
          f'var_loss = {sum(var_loss)/batch_num:.2f}')

