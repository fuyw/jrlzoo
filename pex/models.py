import functools
from typing import Any, Callable, Dict, Sequence, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import checkpoints, train_state
from utils import Batch, target_update


###################
# Utils Functions #
###################
LOG_STD_MIN = -5.
LOG_STD_MAX = 2.


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


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


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


#######################
# Actor-Critic Models #
#######################
class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    std_temperature: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim,
                                 kernel_init=init_fn(self.initializer, 5 / 3))
        self.log_std = self.param('log_std', nn.initializers.zeros,
                                  (self.act_dim, ))

    def __call__(self, rng: Any, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        x = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        mean_action = nn.tanh(x) * self.max_action
        action_distribution = distrax.MultivariateNormalDiag(
            mean_action, self.std_temperature * std)
        sampled_action = action_distribution.sample(seed=rng)
        return mean_action, sampled_action

    def get_logp(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        x = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        mean_action = nn.tanh(x) * self.max_action
        action_distribution = distrax.MultivariateNormalDiag(mean_action, std)
        logp = action_distribution.log_prob(actions)
        return logp


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

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
        qs = VmapCritic(self.hidden_dims, self.initializer)(observations,
                                                            actions)
        return qs


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.out_layer = nn.Dense(1,
                                  kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        v = self.out_layer(x)
        return v.squeeze(-1)


#############
# PEX Agent #
#############
class PEXAgent:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float = 1.0,
                 hidden_dims: Sequence[int] = (256, 256),
                 seed: int = 42,
                 lr: float = 3e-4,
                 tau: float = 0.05,
                 expectile: float = 0.7,
                 std_temperature: float = 1.0,
                 adv_temperature: float = 3.0,
                 max_timesteps: int = int(1e6),
                 initializer: str = "orthogonal"):

        self.act_dim = act_dim
        self.expectile = expectile
        self.adv_temperature = adv_temperature
        self.tau = tau
        self.lr = lr
        self.max_action = max_action

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key, value_key = jax.random.split(self.rng, 4)
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # freezed offline actor
        self.offline_actor = Actor(act_dim=act_dim,
                                   max_action=max_action,
                                   std_temperature=std_temperature,
                                   hidden_dims=hidden_dims,
                                   initializer=initializer)
        offline_actor_params = self.offline_actor.init(
            actor_key, actor_key, dummy_obs)["params"]
        schedule_fn = optax.cosine_decay_schedule(-self.lr, max_timesteps)
        self.offline_actor_state = train_state.TrainState.create(
            apply_fn=self.offline_actor.apply,
            params=offline_actor_params,
            tx=optax.chain(optax.scale_by_adam(),
                           optax.scale_by_schedule(schedule_fn)))

        # online actor
        self.online_actor = Actor(act_dim=act_dim,
                                  max_action=max_action,
                                  std_temperature=std_temperature,
                                  hidden_dims=hidden_dims,
                                  initializer=initializer)
        online_actor_params = self.online_actor.init(
            actor_key, actor_key, dummy_obs)["params"]
        self.online_actor_state = train_state.TrainState.create(
            apply_fn=self.online_actor.apply,
            params=online_actor_params,
            tx=optax.adam(learning_rate=self.lr))

        self.critic = DoubleCritic(hidden_dims=hidden_dims,
                                   initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs,
                                         dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=self.lr))

        self.value = ValueCritic(hidden_dims, initializer=initializer)
        value_params = self.value.init(value_key, dummy_obs)["params"]
        self.value_state = train_state.TrainState.create(
            apply_fn=self.value.apply,
            params=value_params,
            tx=optax.adam(learning_rate=self.lr))

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self,
                       online_actor_params: FrozenDict,
                       offline_actor_params: FrozenDict,
                       critic_params: FrozenDict,
                       rng: Any,
                       observation: np.ndarray) -> jnp.ndarray:
        rng, online_key, offline_key = jax.random.split(rng, 3)
        online_mean_action, online_sampled_action = self.online_actor.apply(
            {"params": online_actor_params}, online_key, observation)
        offline_mean_action, offline_sampled_action = self.offline_actor.apply(
            {"params": offline_actor_params}, offline_key, observation) 
        online_q = self.critic.apply({"params": critic_params}, observation, online_sampled_action).min()
        offline_q = self.critic.apply({"params": critic_params}, observation, offline_sampled_action).min()

        return rng, online_mean_action, online_sampled_action, offline_sampled_action, online_q, offline_q

    def sample_action(self,
                      observation: np.ndarray,
                      eval_mode: bool = False) -> np.ndarray:
        (self.rng,
         online_mean_action,
         online_sampled_action,
         offline_sampled_action,
         online_q,
         offline_q) = self._sample_action(self.online_actor_state.params,
                                          self.offline_actor_state.params,
                                          self.critic_state.params,
                                          self.rng,
                                          observation)
        if eval_mode:
            action = np.asarray(online_mean_action)
        else:
            sampled_qs = np.array([online_q, offline_q])
            sampled_qs -= max(sampled_qs)
            exp_qs = np.exp(sampled_qs)
            prob = exp_qs/exp_qs.sum()
            idx = np.random.choice(2, p=prob)
            if idx == 0:
                action = np.array(online_sampled_action)
            else:
                action = np.array(offline_sampled_action)
        return action.clip(-self.max_action, self.max_action)

    def value_train_step(
        self, batch: Batch, value_state: train_state.TrainState,
        critic_target_params: FrozenDict
    ) -> Tuple[Dict, train_state.TrainState]:
        q1, q2 = self.critic.apply({"params": critic_target_params},
                                   batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)

        def loss_fn(params: FrozenDict):
            v = self.value.apply({"params": params}, batch.observations)
            weight = jnp.where(q - v > 0, self.expectile, 1 - self.expectile)
            value_loss = weight * jnp.square(q - v)
            avg_value_loss = value_loss.mean()
            return avg_value_loss, {
                "value_loss": avg_value_loss,
                "weight": weight.mean(),
                "v": v.mean(),
            }

        (_, value_info), value_grads = jax.value_and_grad(
            loss_fn, has_aux=True)(value_state.params)
        value_state = value_state.apply_gradients(grads=value_grads)
        return value_info, value_state

    def actor_train_step(
        self, batch: Batch, actor_state: train_state.TrainState,
        value_params: FrozenDict, critic_target_params: FrozenDict
    ) -> Tuple[Dict, train_state.TrainState]:
        v = self.value.apply({"params": value_params}, batch.observations)
        q1, q2 = self.critic.apply({"params": critic_target_params},
                                   batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * self.adv_temperature)
        exp_a = jnp.minimum(exp_a, 100.0)

        def loss_fn(params):
            logp = self.online_actor.apply({"params": params},
                                    batch.observations,
                                    batch.actions,
                                    method=Actor.get_logp)
            actor_loss = -exp_a * logp
            avg_actor_loss = actor_loss.mean()
            return avg_actor_loss, {
                "actor_loss": avg_actor_loss,
                "exp_a": exp_a.mean(),
                "adv": (q - v).mean(),
                "logp": logp.mean(),
                "logp_min": logp.min(),
                "logp_max": logp.max(),
            }

        (_, actor_info), actor_grads = jax.value_and_grad(
            loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        return actor_info, actor_state

    def critic_train_step(
            self, batch: Batch, critic_state: train_state.TrainState,
            value_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        next_v = self.value.apply({"params": value_params},
                                  batch.next_observations)
        target_q = batch.rewards + batch.discounts * next_v

        def loss_fn(params: FrozenDict):
            q1, q2 = self.critic.apply({"params": params}, batch.observations,
                                       batch.actions)
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2
            avg_critic_loss = critic_loss.mean()
            return avg_critic_loss, {
                "critic_loss": avg_critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
                "target_q": target_q.mean(),
            }

        (_, critic_info), critic_grads = jax.value_and_grad(
            loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self, batch: Batch, actor_state: train_state.TrainState,
                   value_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):
        value_info, new_value_state = self.value_train_step(
            batch, value_state, critic_target_params)
        actor_info, new_actor_state = self.actor_train_step(
            batch, actor_state, new_value_state.params, critic_target_params)
        critic_info, new_critic_state = self.critic_train_step(
            batch, critic_state, new_value_state.params)
        new_critic_target_params = target_update(new_critic_state.params,
                                                 critic_target_params,
                                                 self.tau)
        return new_actor_state, new_value_state, new_critic_state, new_critic_target_params, {
            **actor_info,
            **value_info,
            **critic_info
        }

    def update(self, batch: Batch):
        (self.online_actor_state, self.value_state, self.critic_state,
         self.critic_target_params,
         log_info) = self.train_step(batch, self.online_actor_state, self.value_state,
                                     self.critic_state,
                                     self.critic_target_params)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname,
                                    self.online_actor_state,
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
        checkpoints.save_checkpoint(fname,
                                    self.value_state,
                                    cnt,
                                    prefix="value_",
                                    keep=20,
                                    overwrite=True)

    def load(self, ckpt_dir, step):
        self.offline_actor_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.offline_actor_state,
            step=step,
            prefix="actor_")
        self.online_actor_state = train_state.TrainState.create(
            apply_fn=self.online_actor.apply,
            params=self.offline_actor_state.params,
            tx=optax.adam(learning_rate=self.lr))  # remove lr scheduler
        self.critic_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.critic_state,
            step=step,
            prefix="critic_")
        self.value_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.value_state,
            step=step,
            prefix="value_")
        self.critic_target_params = self.critic_state.params

    def logger(self, t, logger, log_info):
        logger.info(
            f"\n[#Step {t}] eval_reward: {log_info['reward']:.2f}, eval_time: {log_info['eval_time']:.2f}, time: {log_info['time']:.2f}\n"
            f"\tcritic_loss: {log_info['critic_loss']:.3f}, actor_loss: {log_info['actor_loss']:.3f}, value_loss: {log_info['value_loss']:.3f}\n"
            f"\tq1: {log_info['q1']:.3f}, q2: {log_info['q2']:.3f}, target_q: {log_info['target_q']:.3f}\n"
            f"\tlogp: {log_info['logp']:.3f}, logp_min: {log_info['logp_min']:.3f}, logp_max: {log_info['logp_max']:.3f}\n"
            f"\tadv: {log_info['adv']:.3f}, exp_a: {log_info['exp_a']:.3f}, weight: {log_info['adv']:.3f}, v: {log_info['v']:.3f}\n"
            f"\tbuffer_size: {log_info['buffer_size']/1000:.0f}, buffer_ptr: {log_info['buffer_ptr']/1000:.0f}, "
            f"sample_age: {log_info['sample_age']/1000:.0f}, online_ratio: {log_info['online_ratio']:.3f}\n"
        )

