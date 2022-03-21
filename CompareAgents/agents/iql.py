from typing import Callable, Dict, Sequence, Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import collections
import distrax
import functools
import jax
import jax.numpy as jnp
import optax


LOG_STD_MAX = 2.
LOG_STD_MIN = -5.


def init_fn(initializer: str, gain: float = jnp.sqrt(2)):
    if initializer == "orthogonal":
        return nn.initializers.orthogonal(gain)
    elif initializer == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    elif initializer == "glorot_normal":
        return nn.initializers.glorot_normal()
    return nn.initializers.lecun_normal()


Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


def target_update(params: FrozenDict, target_params: FrozenDict, tau: float) -> FrozenDict:
    def _update(param: FrozenDict, target_param: FrozenDict):
        return tau*param + (1-tau)*target_param
    updated_params = jax.tree_multimap(_update, params, target_params)
    return updated_params


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
    initializer: str = "orthogonal"

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
    initializer: str = "orthogonal"

    def setup(self):
        self.critic1 = Critic(self.hidden_dims, initializer=self.initializer)
        self.critic2 = Critic(self.hidden_dims, initializer=self.initializer)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2

    def encode(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        embedding = self.critic1.encode(observations, actions)
        return embedding


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(self.initializer), activate_final=True)
        self.out_layer = nn.Dense(1, kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        v = self.out_layer(x)
        return v.squeeze(-1)


class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    temperature: float = 3.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(self.initializer), activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim, kernel_init=init_fn(self.initializer, 5/3))
        self.log_std = self.param('log_std', nn.initializers.zeros, (self.act_dim,))

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        mu = self.mu_layer(x)
        mean_action = nn.tanh(mu) * self.max_action
        return mean_action

    def get_logp(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        mu = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        mean_action = nn.tanh(mu) * self.max_action
        action_distribution = distrax.MultivariateNormalDiag(mean_action, std*self.temperature)
        logp = action_distribution.log_prob(actions)
        return logp


class IQLAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float = 1.0,
                 hidden_dims: Sequence[int] = (256, 256),
                 seed: int = 42,
                 lr: float = 3e-4,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 expectile: float = 0.7,
                 temperature: float = 3.0,
                 max_timesteps: int = 1000000,
                 initializer: str = "orthogonal"):

        self.act_dim = act_dim
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.tau = tau

        rng = jax.random.PRNGKey(seed)
        actor_rng, critic_rng, value_rng = jax.random.split(rng, 3)
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        self.actor = Actor(act_dim=act_dim,
                           max_action=max_action,
                           temperature=temperature,
                           hidden_dims=hidden_dims,
                           initializer=initializer)
        actor_params = self.actor.init(actor_rng, dummy_obs)["params"]
        schedule_fn = optax.cosine_decay_schedule(-lr, max_timesteps)
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)))

        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(critic_rng, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=DoubleCritic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=lr))

        self.value = ValueCritic(hidden_dims, initializer=initializer)
        value_params = self.value.init(value_rng, dummy_obs)["params"]
        self.value_state = train_state.TrainState.create(
            apply_fn=ValueCritic.apply,
            params=value_params,
            tx=optax.adam(learning_rate=lr))

    @functools.partial(jax.jit, static_argnames=("self"), device=jax.devices("cpu")[0])
    def sample_action(self, params: FrozenDict, observation: jnp.ndarray) -> jnp.ndarray:
        sampled_action = self.actor.apply({"params": params}, observation)
        return sampled_action

    def value_train_step(self,
                         batch: Batch,
                         value_state: train_state.TrainState,
                         critic_target_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        q1, q2 = self.critic.apply({"params": critic_target_params}, batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)
        def loss_fn(params: FrozenDict):
            v = self.value.apply({"params": params}, batch.observations)
            weight = jnp.where(q-v>0, self.expectile, 1-self.expectile)
            value_loss = (weight * jnp.square(q-v)).mean()
            return value_loss, {"value_loss": value_loss, "v": v.mean(), "weight": weight.mean()}
        (_, value_info), value_grads = jax.value_and_grad(loss_fn, has_aux=True)(value_state.params)
        value_state = value_state.apply_gradients(grads=value_grads)
        return value_info, value_state

    def actor_train_step(self,
                         batch: Batch,
                         actor_state: train_state.TrainState,
                         value_params: FrozenDict,
                         critic_target_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        v = self.value.apply({"params": value_params}, batch.observations)
        q1, q2 = self.critic.apply({"params": critic_target_params}, batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * self.temperature)
        exp_a = jnp.minimum(exp_a, 100.0)
        def loss_fn(params):
            logp = self.actor.apply({"params": params},
                                    batch.observations,
                                    batch.actions,
                                    method=Actor.get_logp)
            actor_loss = -(exp_a * logp).mean()
            return actor_loss, {"actor_loss": actor_loss, "adv": (q-v).mean(), "logp": logp.mean()}
        (_, actor_info), actor_grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        return actor_info, actor_state

    def critic_train_step(self,
                          batch: Batch,
                          critic_state: train_state.TrainState,
                          value_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        next_v = self.value.apply({"params": value_params}, batch.next_observations)
        target_q = batch.rewards + self.gamma * batch.discounts * next_v
        def loss_fn(params: FrozenDict):
            q1, q2 = self.critic.apply({"params": params}, batch.observations, batch.actions)
            
            # critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2
            real_critic_loss = critic_loss[:128].mean()
            model_critic_loss = critic_loss[128:].mean()
            critic_loss = critic_loss.mean()
            return critic_loss, {"critic_loss": critic_loss,
                                 "real_critic_loss": real_critic_loss,
                                 "model_critic_loss": model_critic_loss,
                                 "q1": q1.mean(),
                                 "q2": q2.mean(),
                                 "target_q": target_q.mean()}
        (_, critic_info), critic_grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        return critic_info, critic_state
    
    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   actor_state: train_state.TrainState,
                   value_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):
        value_info, new_value_state = self.value_train_step(batch, value_state, critic_target_params)
        actor_info, new_actor_state = self.actor_train_step(batch, actor_state, new_value_state.params, critic_target_params)
        critic_info, new_critic_state = self.critic_train_step(batch, critic_state, new_value_state.params)
        new_critic_target_params = target_update(new_critic_state.params, critic_target_params, self.tau)
        return new_actor_state, new_value_state, new_critic_state, new_critic_target_params, {
            **actor_info, **value_info, **critic_info}

    def update(self, batch: Batch):
        (self.actor_state, self.value_state, self.critic_state,
         self.critic_target_params, log_info) = self.train_step(batch, self.actor_state, self.value_state,
                                                                self.critic_state, self.critic_target_params)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.actor_state, cnt, prefix="actor_", keep=110, overwrite=True)
        checkpoints.save_checkpoint(fname, self.critic_state, cnt, prefix="critic_", keep=110, overwrite=True)

    def load(self, ckpt_dir, step):
        self.actor_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.actor_state,
                                                          step=step, prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.critic_state,
                                                           step=step, prefix="critic_")        
        

