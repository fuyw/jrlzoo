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
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.out_layer = nn.Dense(1,
                                  kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze(-1)

    def encode(self, observations: jnp.ndarray,
               actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"
    num_qs: int = 2

    @nn.compact
    def __call__(self, observations, actions):
        VmapCritic = nn.vmap(Critic,
                             variable_axes={"params": 0},
                             split_rngs={"params": True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims, self.initializer)(observations, actions)
        return qs


class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim,
                                 kernel_init=init_fn(self.initializer, 5/3))
        self.std_layer = nn.Dense(self.act_dim,
                                  kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, rng: Any, observation: jnp.ndarray):
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        mean_action = nn.tanh(mu)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))
        sampled_action, logp = action_distribution.sample_and_log_prob(seed=rng)
        return mean_action * self.max_action, sampled_action * self.max_action, logp

    def get_logp(self, observation: jnp.ndarray,
                 action: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        action_distribution = distrax.Normal(mu, std)
        raw_action = atanh(action)
        logp = action_distribution.log_prob(raw_action).sum(-1)
        logp -= 2 * (jnp.log(2) - raw_action -
                     jax.nn.softplus(-2 * raw_action)).sum(-1)
        return logp


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


class SACAgent:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float = 1.0,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 target_entropy: float = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 initializer: str = "glorot_uniform"):

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        if target_entropy is None:
            self.target_entropy = -act_dim / 2  # dopamine setting
        else:
            self.target_entropy = target_entropy

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
        self.actor_state = train_state.TrainState.create(apply_fn=self.actor.apply,
                                                         params=actor_params,
                                                         tx=optax.adam(lr))

        # Initialize the Critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims,
                                   initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply, params=critic_params, tx=optax.adam(lr))

        # Entropy tuning
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.adam(lr))

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self,
                       params: FrozenDict,
                       rng: Any,
                       observation: np.ndarray) -> jnp.ndarray:
        mean_action, sampled_action, _ = self.actor.apply({"params": params}, rng, observation)
        return mean_action, sampled_action

    def sample_action(self, observation: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        self.rng, sample_rng = jax.random.split(self.rng)
        mean_action, sampled_action = self._sample_action(self.actor_state.params, sample_rng, observation)
        action = mean_action if eval_mode else sampled_action
        action = np.asarray(action)
        return action.clip(-self.max_action, self.max_action)

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   key: Any,
                   alpha_state: train_state.TrainState,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        def loss_fn(alpha_params: FrozenDict, actor_params: FrozenDict,
                    critic_params: FrozenDict, rng: Any,
                    observation: jnp.ndarray, action: jnp.ndarray,
                    reward: jnp.ndarray, next_observation: jnp.ndarray,
                    discount: jnp.ndarray):
            """compute loss for a single transition"""
            rng1, rng2 = jax.random.split(rng, 2)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply(
                {"params": actor_params}, rng1, observation)

            # Alpha loss: stop gradient to avoid affecting Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(
                logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha)

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply(
                {"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.minimum(sampled_q1, sampled_q2)

            # Actor loss
            actor_loss = alpha * logp - sampled_q

            # Critic loss
            q1, q2 = self.critic.apply(
                {"params": critic_params}, observation, action)

            # Use frozen_actor_params to avoid affect Actor parameters
            _, next_action, logp_next_action = self.actor.apply(
                {"params": frozen_actor_params}, rng2, next_observation)
            next_q1, next_q2 = self.critic.apply(
                {"params": critic_target_params}, next_observation, next_action)
            next_q = jnp.minimum(next_q1, next_q2) - alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q
            critic_loss1 = (q1 - target_q)**2
            critic_loss2 = (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2

            # Loss weight form Dopamine
            total_loss = critic_loss + actor_loss + alpha_loss
            log_info = {
                "critic_loss1": critic_loss1,
                "critic_loss2": critic_loss2,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "q1": q1,
                "q2": q2,
                "target_q": target_q,
                "alpha": alpha,
                "logp": logp
            }

            return total_loss, log_info

        grad_fn = jax.vmap(jax.value_and_grad(loss_fn,
                                              argnums=(0, 1, 2),
                                              has_aux=True),
                           in_axes=(None, None, None, 0, 0, 0, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))

        (_, log_info), grads = grad_fn(alpha_state.params, actor_state.params,
                                       critic_state.params, keys,
                                       batch.observations, batch.actions,
                                       batch.rewards, batch.next_observations,
                                       batch.discounts)
        grads = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), grads)
        log_info = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), log_info)

        # Update TrainState
        alpha_grads, actor_grads, critic_grads = grads
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)
        new_critic_target_params = target_update(new_critic_state.params,
                                                 critic_target_params,
                                                 self.tau)

        return new_alpha_state, new_actor_state, new_critic_state, new_critic_target_params, log_info

    def update(self, batch: Batch):
        self.rng, key = jax.random.split(self.rng, 2)
        self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params, log_info = self.train_step(
            batch, key, self.alpha_state, self.actor_state, self.critic_state,
            self.critic_target_params)
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
