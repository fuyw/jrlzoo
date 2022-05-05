import jax
import jax.numpy as jnp
from models import GaussianMLP, GaussianMLP2

env_names = []
model = GaussianMLP(7, 18)
rng = jax.random.PRNGKey(0)
dummy_inputs = jnp.ones([1, 18])

