from flax import linen as nn
import jax.numpy as jnp


class ActorCritic(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        """Define the convolutional network architecture.

        Network is used to both estimate policy (logits) and expected state value;
        in other words, hidden layers' params are shared between policy and value
        networks.
        """

        x = x.astype(jnp.float32) / 255.
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name="conv1", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name="conv2", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name="conv3", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape, -1))
        x = nn.Dense(features=512, name="hidden", dtype=jnp.float32)(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.num_outputs, name="logits", dtype=jnp.float32)(x)
        policy_log_probabilities = nn.log_softmax(logits)
        value = nn.Dense(features=1, name="value", dtype=dtype)(x)
        return policy_log_probabilities, value

