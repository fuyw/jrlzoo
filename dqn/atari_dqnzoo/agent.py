import jax.numpy as jnp


class DQNAgent:
    """Deep Q-Network agent."""

    def __init__(
        self,
        replay_buffer,
        batch_size: int,
        seed: int,
    ):
        self._replay_buffer = replay_buffer

        # Initialize

        def loss_fn():
            q_tm1 = network.apply()
