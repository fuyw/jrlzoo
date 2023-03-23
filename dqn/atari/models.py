import functools
import optax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints


class QNetwork_Nature(nn.Module):
    """NatureDQN, faster fps ~800"""
    act_dim: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name="conv1")
        self.conv2 = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name="conv2")
        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name="conv3")
        self.fc_layer = nn.Dense(features=512, name="fc")
        self.out_layer = nn.Dense(features=self.act_dim, name="out")

    def __call__(self, observation):
        x = observation.astype(jnp.float32) / 255.  # (84, 84, 4)
        x = nn.relu(self.conv1(x))                  # (21, 21, 32)
        x = nn.relu(self.conv2(x))                  # (11, 11, 64)
        x = nn.relu(self.conv3(x))                  # (11, 11, 64)
        x = x.reshape(len(observation), -1)         # (7744,)
        x = nn.relu(self.fc_layer(x))               # (512,)
        Qs = self.out_layer(x)                      # (act_dim,)
        return Qs


class QNetwork_CNN4(nn.Module):
    """Better performance, lower fps ~600"""
    act_dim: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(5, 5), strides=(1, 1),
                             padding=(2, 2), name="conv1")
        self.conv2 = nn.Conv(features=32, kernel_size=(5, 5), strides=(1, 1),
                             padding=(2, 2), name="conv2")
        self.conv3 = nn.Conv(features=64, kernel_size=(4, 4), strides=(1, 1),
                             padding=(1, 1), name="conv3")
        self.conv4 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                             padding=(1, 1), name="conv4")
        self.fc_layer = nn.Dense(features=512, name="fc")
        self.out_layer = nn.Dense(features=self.act_dim, name="out")

    def __call__(self, observation):
        x = observation.astype(jnp.float32) / 255.               # (1, 84, 84, 32)
        x = nn.relu(self.conv1(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (1, 42, 42, 32)
        x = nn.relu(self.conv2(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (1, 21, 21, 32)
        x = nn.relu(self.conv3(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (1, 10, 10, 64)
        x = x.reshape(len(observation), -1)                      # (6400,)
        x = nn.relu(self.fc_layer(x))                            # (512,)
        Qs = self.out_layer(x)                                   # (act_dim,)
        return Qs


class DQNAgent:
    def __init__(self,
                 act_dim,
                 seed: int = 42,
                 lr_start: float = 3e-4,
                 lr_end: float = 1e-5,
                 gamma: float = 0.99,
                 total_timesteps: int = int(1e7)):
        self.gamma = gamma
        rng = jax.random.PRNGKey(seed)
        self.q_network = QNetwork_Nature(act_dim)
        params = self.q_network.init(rng, jnp.ones(shape=(1, 84, 84, 4)))["params"]
        self.target_params = params
        self.lr_scheduler = optax.linear_schedule(
            init_value=lr_start, end_value=lr_end, transition_steps=total_timesteps)
        self.state = train_state.TrainState.create(
            apply_fn=self.q_network.apply, params=params,
            tx=optax.adam(self.lr_scheduler))
        self.cnt = 0

    @functools.partial(jax.jit, static_argnums=0)
    def _sample(self, params, observation):
        Qs = self.q_network.apply({"params": params}, observation)
        action = Qs.argmax(-1)
        return action

    def sample(self, observation):
        action = self._sample(self.state.params, observation)
        return action.item()

    @functools.partial(jax.jit, static_argnums=0)
    def train_step(self, state, target_params, batch):
        next_Q = self.q_network.apply({"params": target_params}, batch.next_observations).max(-1)
        target_Q = batch.rewards + self.gamma * next_Q * batch.discounts
        def loss_fn(params):
            Qs = self.q_network.apply({"params": params}, batch.observations)
            Q = jax.vmap(lambda q,a: q[a])(Qs, batch.actions)
            loss = (Q - target_Q) ** 2
            log_info = {
                "avg_Q": Q.mean(),
                "avg_target_Q": target_Q.mean(),
                "avg_loss": loss.mean(),
            }
            return loss.mean(), log_info
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, log_info

    def update(self, batch):
        self.state, log_info = self.train_step(self.state, self.target_params, batch)
        return log_info

    def sync_target_network(self):
        self.target_params = self.state.params

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.state, cnt, prefix="qnet_", keep=20, overwrite=True)
