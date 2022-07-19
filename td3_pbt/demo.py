from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
from utils import target_update, Batch


def init_fn(initializer: str, gain: float = jnp.sqrt(2)):
    if initializer == "orthogonal":
        return nn.initializers.orthogonal(gain)
    elif initializer == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    elif initializer == "glorot_normal":
        return nn.initializers.glorot_normal()
    return nn.initializers.lecun_normal()


class VectorizedDense(nn.Module):
    """Vectorized linear layer.

    Input/Output shapes:
        inputs.shape = (population_num, input_dim)
        y.shape      = (population_num, output_dim)
    """

    population_num: int
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.array) -> jnp.array:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param("kernel", self.kernel_init,
                            (self.population_num, inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.einsum("ij,ijk->ik", inputs, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.population_num, self.features))
            bias = jnp.asarray(bias, self.dtype)
            y += bias
        return y


class VectorizedMLP(nn.Module):
    """Vectorized MLP."""

    population_num: int
    hidden_dims: Sequence[int] = (256, 256)
    init_fn: Callable = nn.initializers.glorot_uniform()
    activate_final: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = VectorizedDense(population_num=self.population_num,
                                features=size,
                                kernel_init=self.init_fn)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


class VectorizedCritic(nn.Module):
    """Vectorized Critic for jax.vmap

    Shapes:
        Input x.shape  = (population_num, obs_dim+act_dim)
        Output q.shape = (population_num,)
    """
    population_num: int
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = VectorizedMLP(population_num=self.population_num,
                                 hidden_dims=self.hidden_dims,
                                 init_fn=init_fn(self.initializer),
                                 activate_final=True)
        self.out_layer = VectorizedDense(population_num=self.population_num,
                                         features=1,
                                         kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)   # (population_num, 1)
        return q.squeeze(-1)


class VectorizedDoubleCritic(nn.Module):
    population_num: int = 10
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.critic1 = VectorizedCritic(population_num=self.population_num,
                                        hidden_dims=self.hidden_dims,
                                        initializer=self.initializer)
        self.critic2 = VectorizedCritic(population_num=self.population_num,
                                        hidden_dims=self.hidden_dims,
                                        initializer=self.initializer)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2

    def Q1(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observations, actions)
        return q1


class VectorizedActor(nn.Module):
    population_num: int
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = VectorizedMLP(population_num=self.population_num,
                                 hidden_dims=self.hidden_dims,
                                 init_fn=init_fn(self.initializer),
                                 activate_final=True)
        self.out_layer = VectorizedDense(population_num=self.population_num,
                                         features=self.act_dim,
                                         kernel_init=init_fn(self.initializer, 1e-2))

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        x = self.out_layer(x)
        mean_action = nn.tanh(x) * self.max_action
        return mean_action



seed = 42
population_num = 10
obs_dim, act_dim = 13, 7
rng = jax.random.PRNGKey(seed)
rng, actor_rng, critic_rng = jax.random.split(rng, 3)
dummy_obs = jnp.ones([population_num, obs_dim], dtype=jnp.float32)
dummy_act = jnp.ones([population_num, act_dim], dtype=jnp.float32)
test_obs = jax.random.normal(rng, [population_num, obs_dim])
test_act = jax.random.normal(rng, [population_num, act_dim])


critic = VectorizedCritic(population_num=population_num)
critic_params = critic.init(critic_rng, dummy_obs, dummy_act)["params"]
critic_state = train_state.TrainState.create(apply_fn=VectorizedCritic.apply,
                                             params=critic_params,
                                             tx=optax.adam(1e-3))


actor = VectorizedActor(population_num=population_num, act_dim=act_dim)
actor_params = actor.init(actor_rng, dummy_obs)["params"]
actor_state = train_state.TrainState.create(apply_fn=VectorizedActor.apply,
                                            params=actor_params,
                                            tx=optax.adam(1e-3))


outputs = actor.apply({"params": actor_params}, obs)  # (10, 7)

