from flax import linen as nn
import jax.numpy as jnp


DTYPE = jnp.float32


class ActorCritic(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x):
        """Define the CNN architecture."""
        x = x.astype(jnp.float32) / 255.
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1',
                    dtype=DTYPE)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2',
                    dtype=DTYPE)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv3',
                    dtype=DTYPE)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=512, name='hidden', dtype=DTYPE)(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.out_dim, name='logits', dtype=DTYPE)(x)
        policy_log_probabilities = nn.log_softmax(logits)
        value = nn.Dense(features=1, name='value', dtype=DTYPE)(x)
        return policy_log_probabilities, value   
