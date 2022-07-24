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
    """Vectorized Actor for jax.vmap

    Shapes:
        Input x.shape  = (population_num, obs_dim)
        Output x.shape = (population_num, act_dim)
    """
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


class VectorizedTD3Agent:
    def __init__(self,
                 population_num: int,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float = 1.0,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 noise_clip: float = 0.5,
                 policy_noise: float = 0.2,
                 policy_freq: int = 2,
                 lr: float = 3e-4,
                 alpha: float = 2.5,
                 seed: int = 42,
                 hidden_dims: Sequence[int] = (256, 256),
                 initializer: str = "glorot_uniform"):

        self.max_action = max_action
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        rng = jax.random.PRNGKey(seed)
        self.actor_rng, self.critic_rng = jax.random.split(rng, 2)

        # Dummy inputs
        dummy_obs = jnp.ones([population_num, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([population_num, act_dim], dtype=jnp.float32)

        self.actor = VectorizedActor(population_num=population_num,
                                     act_dim=act_dim,
                                     max_action=max_action,
                                     hidden_dims=hidden_dims,
                                     initializer=initializer)
        actor_params = self.actor.init(self.actor_rng, dummy_obs)["params"]
        self.actor_target_params = actor_params
        self.actor_state = train_state.TrainState.create(apply_fn=Actor.apply,
                                                         params=actor_params,
                                                         tx=optax.adam(learning_rate=lr))

        # Initialize the critic
        self.critic = VectorizedDoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(self.critic_rng, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(apply_fn=Critic.apply,
                                                          params=critic_params,
                                                          tx=optax.adam(learning_rate=lr))
        self.update_step = 0

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self, params: FrozenDict, observation: np.ndarray) -> jnp.ndarray:
        sampled_action = self.actor.apply({"params": params}, observation)
        return sampled_action

    def sample_action(self, params: FrozenDict, observation: np.ndarray) -> np.ndarray:
        sampled_action = self._sample_action(params, observation)
        sampled_action = np.asarray(sampled_action)
        return sampled_action.clip(-1.0, 1.0)

    def actor_train_step(self, batch: Batch, actor_state: train_state.TrainState, critic_params: FrozenDict):
        def loss_fn(params: FrozenDict):
            actions = self.actor.apply({"params": params}, batch.observations)
            q = self.critic.apply({"params": critic_params}, batch.observations, actions, method=DoubleCritic.Q1)
            actor_loss = -q
            avg_actor_loss = actor_loss.mean()
            return avg_actor_loss, {
                "actor_loss": avg_actor_loss,
                "max_actor_loss": actor_loss.max(),
                "min_actor_loss": actor_loss.min(),
            }
        (_, actor_info), actor_grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        return actor_info, actor_state

    def critic_train_step(self,
                          batch: Batch,
                          critic_key: Any,
                          critic_state: train_state.TrainState,
                          actor_target_params: FrozenDict,
                          critic_target_params: FrozenDict):

        noise = jax.random.normal(critic_key, batch.actions.shape) * self.policy_noise
        noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)
        next_actions = self.actor.apply({"params": actor_target_params}, batch.next_observations)  # (B, act_dim)
        next_actions = jnp.clip(next_actions + noise, -self.max_action, self.max_action)

        next_q1, next_q2 = self.critic.apply({"params": critic_target_params}, batch.next_observations, next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        target_q = batch.rewards + self.gamma * batch.discounts * next_q
        def loss_fn(params: FrozenDict):
            q1, q2 = self.critic.apply({"params": params}, batch.observations, batch.actions)
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2
            avg_critic_loss = critic_loss.mean()

            return avg_critic_loss, {
                "critic_loss": avg_critic_loss, "max_critic_loss": critic_loss.max(), "min_critic_loss": critic_loss.min(),
                "target_q": target_q.mean(), "max_target_q": target_q.max(), "min_target_q": target_q.min(),
                "q1": q1.mean(), "max_q1": q1.max(), "min_q1": q1.min(),
                "q2": q2.mean(), "max_q2": q2.max(), "min_q2": q2.min(),
            }
        (_, critic_info), critic_grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnames=("self", "delay_update"))
    def train_step(self,
                   batch: Batch,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   actor_target_params: FrozenDict,
                   critic_target_params: FrozenDict,
                   critic_key: Any,
                   delay_update: bool):
        critic_info, new_critic_state = self.critic_train_step(
            batch, critic_key, critic_state, actor_target_params, critic_target_params)
        if delay_update:
            actor_info, new_actor_state = self.actor_train_step(batch, actor_state, critic_state.params)
            params = (new_actor_state.params, new_critic_state.params)
            target_params = (actor_target_params, critic_target_params)
            new_actor_target_params, new_critic_target_params = target_update(params, target_params, self.tau)
            return new_actor_state, new_critic_state, new_actor_target_params, new_critic_target_params, {**actor_info, **critic_info}
        return new_critic_state, critic_info

    def update(self, batch: Batch):
        self.update_step += 1
        self.critic_rng, critic_key = jax.random.split(self.critic_rng, 2)
        if self.update_step % 2 == 0:
            (self.actor_state, self.critic_state, self.actor_target_params, self.critic_target_params,
             log_info) = self.train_step(batch, self.actor_state, self.critic_state, self.actor_target_params,
                self.critic_target_params, critic_key, True)
        else:
            self.critic_state, log_info = self.train_step(batch, self.actor_state, self.critic_state,
                self.actor_target_params, self.critic_target_params, critic_key, False)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)

    def load(self, fname, step):
        self.actor_state = checkpoints.restore_checkpoint(ckpt_dir=fname, target=self.actor_state, step=step, prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(ckpt_dir=fname, target=self.critic_state, step=step, prefix="critic_")