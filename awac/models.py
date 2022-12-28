from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import distrax
import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
from utils import target_update, Batch

LOG_STD_MAX = 2.
LOG_STD_MIN = -10.


def init_fn(initializer: str, gain: float = jnp.sqrt(2)):
    if initializer == "orthogonal":
        return nn.initializers.orthogonal(gain)
    elif initializer == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    elif initializer == "glorot_normal":
        return nn.initializers.glorot_normal()
    return nn.initializers.lecun_normal()


def atanh(x: jnp.ndarray):
    one_plus_x = jnp.clip(1 + x, a_min=1e-6)
    one_minus_x = jnp.clip(1 - x, a_min=1e-6)
    return 0.5 * jnp.log(one_plus_x / one_minus_x)


class MLP(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    init_fn: Callable = nn.initializers.glorot_uniform()
    activate_final: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.init_fn)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "glorot_uniform"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(self.initializer), activate_final=True)
        self.out_layer = nn.Dense(1, kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze(-1)

    def encode(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "glorot_uniform"

    def setup(self):
        self.critic1 = Critic(self.hidden_dims, initializer=self.initializer)
        self.critic2 = Critic(self.hidden_dims, initializer=self.initializer)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2

    def Q1(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observations, actions)
        return q1


class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    temperature: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(self.initializer), activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim, kernel_init=init_fn(self.initializer, 5/3))
        self.log_std = self.param('log_std', nn.initializers.zeros, (self.act_dim,))

    def __call__(self, rng: Any, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        x = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        mean_action = nn.tanh(x) * self.max_action
        action_distribution = distrax.MultivariateNormalDiag(mean_action, std*self.temperature)
        sampled_action, logp = action_distribution.sample_and_log_prob(seed=rng)
        return mean_action, sampled_action, logp

    # without tanh
    def get_logp(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        x = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        mean_action = nn.tanh(x) * self.max_action
        action_distribution = distrax.MultivariateNormalDiag(mean_action, std*self.temperature)
        log_prob = action_distribution.log_prob(actions)
        return log_prob


class AWACAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float = 1.0,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 temperature: float = 0.5,
                 hidden_dims: Sequence[int] = (256, 256),
                 initializer: str = "glorot_uniform"):

        self.gamma = gamma
        self.tau = tau
        self.temperature = temperature

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(act_dim=act_dim,
                           max_action=max_action,
                           hidden_dims=hidden_dims,
                           initializer=initializer)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(apply_fn=Actor.apply,
                                                         params=actor_params,
                                                         tx=optax.adam(lr))

        # Initialize the Critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=Critic.apply, params=critic_params, tx=optax.adam(lr))

    @functools.partial(jax.jit, static_argnames=("self", "eval_mode"))
    def _sample_action(self, params: FrozenDict, rng: Any, observation: np.ndarray, eval_mode: bool = False) -> jnp.ndarray:
        rng, sample_rng = jax.random.split(rng)
        mean_action, sampled_action, _ = self.actor.apply({"params": params}, sample_rng, observation)
        return rng, jnp.where(eval_mode, mean_action, sampled_action)

    def sample_action(self, params: FrozenDict, rng: Any, observation: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        rng, sampled_action = self._sample_action(params, rng, observation, eval_mode)
        sampled_action = np.asarray(sampled_action)
        return rng, sampled_action.clip(-1.0, 1.0)

    def actor_train_step(self,
                         batch: Batch,
                         actor_key: Any,
                         actor_state: train_state.TrainState,
                         critic_params: FrozenDict):
        q1, q2 = self.critic.apply({"params": critic_params}, batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)

        def loss_fn(actor_params: FrozenDict):
            _, sampled_actions, _ = self.actor.apply({"params": actor_params}, actor_key, batch.observations)
            sampled_q1, sampled_q2 = self.critic.apply(
                {"params": critic_params}, batch.observations, sampled_actions)
            v = jnp.minimum(sampled_q1, sampled_q2)

            logp = self.actor.apply({"params": actor_params},
                                    batch.observations,
                                    batch.actions,
                                    method=Actor.get_logp)
            exp_a = jnp.exp((q - v) * self.temperature)
            exp_a = jnp.minimum(exp_a, 100.0)

            # actor_loss = -exp_a * logp

            actor_loss = -jax.nn.softmax((q - v)/2.0) * logp

            log_info = {
                "actor_loss": actor_loss.sum(),
                "actor_loss_max": actor_loss.max(),
                "actor_loss_min": actor_loss.min(),
                "actor_loss_std": actor_loss.std(),
                "exp_a": exp_a.mean(),
                "exp_a_max": exp_a.max(),
                "exp_a_min": exp_a.min(),
                "v": v.mean(),
                "v_max": v.max(),
                "v_min": v.min(),
                "logp": logp.mean(),
                "logp_max": logp.max(),
                "logp_min": logp.min(),
            }
            return actor_loss.sum(), log_info

        (_, log_info), actor_grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        return log_info, new_actor_state

    def critic_train_step(self,
                          batch: Batch,
                          critic_key: Any,
                          critic_state: train_state.TrainState,
                          actor_params: FrozenDict,
                          critic_target_params: FrozenDict):

        def loss_fn(critic_params: FrozenDict):
            q1, q2 = self.critic.apply({"params": critic_params}, batch.observations, batch.actions)
            _, next_actions, _ = self.actor.apply({"params": actor_params}, critic_key, batch.next_observations)
            next_q1, next_q2 = self.critic.apply(
                {"params": critic_target_params}, batch.next_observations, next_actions)
            next_q = jnp.minimum(next_q1, next_q2)
            target_q = batch.rewards + self.gamma * batch.discounts * next_q
            critic_loss1 = (q1 - target_q)**2
            critic_loss2 = (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2

            # Total loss
            log_info = {
                "critic_loss1": critic_loss1.mean(),
                "critic_loss1_min": critic_loss1.min(),
                "critic_loss1_max": critic_loss1.max(),
                "critic_loss2": critic_loss2.mean(),
                "critic_loss2_min": critic_loss2.min(),
                "critic_loss2_max": critic_loss2.max(),
                "critic_loss": critic_loss.mean(),
                "critic_loss_min": critic_loss.min(),
                "critic_loss_max": critic_loss.max(),
                "q1": q1.mean(),
                "q1_min": q1.min(),
                "q1_max": q1.max(),
                "q2": q2.mean(),
                "q2_min": q2.min(),
                "q2_max": q2.max(),
                "target_q": target_q.mean(),
                "target_q_min": target_q.min(),
                "target_q_max": target_q.max(),
            }
            return critic_loss.mean(), log_info

        (_, log_info), critic_grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)
        return log_info, new_critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   actor_key: Any,
                   critic_key: Any,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):
        actor_info, new_actor_state = self.actor_train_step(
            batch, actor_key, actor_state, critic_state.params)
        critic_info, new_critic_state = self.critic_train_step(
            batch, critic_key, critic_state, actor_state.params, critic_target_params)
        new_critic_target_params = target_update(new_critic_state.params, critic_target_params, self.tau)
        return new_actor_state, new_critic_state, new_critic_target_params, {**actor_info, **critic_info}
        # critic_info = {}
        # return new_actor_state, critic_state, critic_target_params, {**actor_info, **critic_info}

    def update(self, batch: Batch):
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)
        self.actor_state, self.critic_state, self.critic_target_params, log_info = self.train_step(
            batch, actor_key, critic_key, self.actor_state,
            self.critic_state, self.critic_target_params)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname,
                                    self.actor_state,
                                    cnt,
                                    prefix="actor_",
                                    keep=20,
                                    overwrite=True)
        checkpoints.save_checkpoint(fname,
                                    self.critic_state,
                                    cnt,
                                    prefix="critic_",
                                    keep=20,
                                    overwrite=True)

    def load(self, ckpt_dir, step):
        self.actor_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.actor_state,
            step=step,
            prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.critic_state,
            step=step,
            prefix="critic_")
