import jax.numpy as jnp
from flax import linen as nn

init_fn = nn.initializers.xavier_uniform()


class QNetwork(nn.Module):
    act_dim: int

    def setup(self):
        self.conv1 = nn.Conv(features=32,
                             kernel_size=(8, 8),
                             strides=(4, 4),
                             name="conv1")
        self.conv2 = nn.Conv(features=64,
                             kernel_size=(4, 4),
                             strides=(2, 2),
                             name="conv2")
        self.conv3 = nn.Conv(features=64,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             name="conv3")
        self.fc_layer = nn.Dense(features=512, name="fc")
        self.out_layer = nn.Dense(features=self.act_dim, name="out")

    def __call__(self, observation):
        x = observation.astype(jnp.float32) / 255.  # (84, 84, 4)
        x = nn.relu(self.conv1(x))  # (21, 21, 32)
        x = nn.relu(self.conv2(x))  # (11, 11, 64)
        x = nn.relu(self.conv3(x))  # (11, 11, 64)
        x = x.reshape(len(observation), -1)  # (7744,)
        x = nn.relu(self.fc_layer(x))  # (512,)
        Qs = self.out_layer(x)  # (act_dim,)
        return Qs
