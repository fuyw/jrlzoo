from typing import Tuple
import optax
import functools
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state


init_fn = nn.initializers.xavier_uniform()
class QNetwork(nn.Module):
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


class DQNAgent:
    def __init__(self,
                 num_actions: int,
                 observation_shape: Tuple[int] = (1, 84, 84, 4),
                 seed: int = 42,
                 lr: float = 3e-4,
                 stack_size: int = 4,
                 gamma: float = 0.99):
        self.gamma = gamma

        # initialize DQN network
        rng = jax.random.PRNGKey(seed)
        self.q_network = QNetwork(num_actions)
        params = self.q_network.init(rng, jnp.ones(observation_shape))["params"]
        self.state = train_state.TrainState.create(apply_fn=self.q_network.apply,
                                                   params=params,
                                                   tx=optax.adam(lr))
        target_params = params


def update(network_def, state, target_params, batch, gamma):
    next_Q = network_def.apply({"params": target_params}, batch.next_observations).max(-1)
    target_Q = batch.rewards + gamma * next_Q * batch.discounts
    def loss_fn(params):
        Qs = network_def.apply({"params": params}, batch.observations)
        Q = jax.vmap(lambda q,a: q[a])(Qs, batch.actions.reshape(-1, 1)).squeeze()


