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
    initializer: str = "glorot_uniform"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim, kernel_init=init_fn(self.initializer, 1e-2))
        self.std_layer = nn.Dense(self.act_dim, kernel_init=init_fn(self.initializer, 1e-2))

    def __call__(self, rng: Any, observation: jnp.ndarray) -> jnp.ndarray:
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
        return mean_action*self.max_action, sampled_action*self.max_action, logp

    def get_logp(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        action_distribution = distrax.Normal(mu, std)
        raw_action = atanh(action)
        logp = action_distribution.log_prob(raw_action).sum(-1)
        logp -= 2*(jnp.log(2) - raw_action - jax.nn.softplus(-2*raw_action)).sum(-1)
        return logp

    def encode(self, observations):
        embeddings = self.net(observations)
        return embeddings


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)
    
    def __call__(self):
        return self.value


class CQLAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float = 1.0,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr_critic: float = 3e-4,
                 lr_actor: float = 1e-4,
                 target_entropy: float = None,
                 backup_entropy: bool = False,
                 num_random: int = 10,
                 min_q_weight: float = 5.0,
                 bc_timesteps: int = 0,
                 with_lagrange: bool = False,
                 lagrange_thresh: float = 10.0,
                 max_target_backup: bool = False,
                 cql_clip_diff_min: float = -np.inf,
                 cql_clip_diff_max: float = np.inf,
                 actor_hidden_dims: Sequence[int] = (256, 256),
                 critic_hidden_dims: Sequence[int] = (256, 256),
                 initializer: str = "glorot_uniform"):

        self.update_step = 0
        self.bc_timesteps = bc_timesteps
        self.max_action = max_action
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.backup_entropy = backup_entropy
        if target_entropy is None:
            self.target_entropy = -act_dim
        else:
            self.target_entropy = target_entropy

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(act_dim=act_dim, max_action=max_action,
                           hidden_dims=actor_hidden_dims, initializer=initializer)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=lr_actor))

        # Initialize the Critic
        self.critic = DoubleCritic(hidden_dims=critic_hidden_dims, initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=lr_critic))

        # Entropy tuning
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.adam(lr_actor)
        )

        # CQL parameters
        self.num_random = num_random
        self.with_lagrange = with_lagrange
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        # self.min_q_weight = 1.0 if with_lagrange else min_q_weight
        self.min_q_weight = min_q_weight
        self.max_target_backup = max_target_backup
        if self.with_lagrange:
            self.lagrange_thresh = lagrange_thresh
            self.rng, cql_key = jax.random.split(self.rng, 2)
            self.log_cql_alpha = Scalar(1.0)
            self.cql_alpha_state = train_state.TrainState.create(
                apply_fn=None,
                params=self.log_cql_alpha.init(cql_key)["params"],
                tx=optax.adam(lr_actor)
            )

    @functools.partial(jax.jit, static_argnames=("self", "eval_mode"))
    def _sample_action(self,
                       params: FrozenDict,
                       rng: Any,
                       observation: np.ndarray,
                       eval_mode: bool = False) -> jnp.ndarray:
        mean_action, sampled_action, _ = self.actor.apply({"params": params}, rng, observation)
        return jnp.where(eval_mode, mean_action, sampled_action)

    def sample_action(self, observation: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        self.rng, sample_rng = jax.random.split(self.rng)
        sampled_action = self._sample_action(self.actor_state.params, sample_rng, observation, eval_mode)
        sampled_action = np.asarray(sampled_action)
        return sampled_action.clip(-self.max_action, self.max_action)

    def actor_alpha_train_step(self, batch, key, alpha_state, actor_state, critic_state, bc):
        frozen_critic_params = critic_state.params
        def loss_fn(alpha_params, actor_params, rng, observation, action):
            _, sampled_action, logp = self.actor.apply({"params": actor_params}, rng, observation)
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha) 
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.minimum(sampled_q1, sampled_q2)
            if bc:
                logp_bc = self.actor.apply({"params": actor_params}, observation, action, method=Actor.get_logp)
                actor_loss = alpha * logp - logp_bc
            else:
                actor_loss = alpha * logp - sampled_q

            # return info
            actor_alpha_loss = actor_loss + alpha_loss
            log_info = {
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "sampled_q": sampled_q,
                "alpha": alpha,
                "logp": logp
            }
            return actor_alpha_loss, log_info

        # compute gradient with vmap
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn,
                                              argnums=(0, 1),
                                              has_aux=True),
                           in_axes=(None, None, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))

        (_, log_info), grads = grad_fn(alpha_state.params,
                                       actor_state.params,
                                       keys,
                                       batch.observations,
                                       batch.actions)
        grads = jax.tree_util.tree_map(
            functools.partial(jnp.mean, axis=0), grads)
        log_info = jax.tree_util.tree_map(
            functools.partial(jnp.mean, axis=0), log_info)

        # Update TrainState
        alpha_grads, actor_grads = grads
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        return new_alpha_state, new_actor_state, log_info

    def critic_train_step(self,
                          batch: Batch,
                          key: Any,
                          actor_state: train_state.TrainState,
                          critic_state: train_state.TrainState,
                          critic_target_params: FrozenDict,
                          sac_alpha: float = 1.0,
                          cql_alpha: float = 1.0):

        # prepare prozen params to avoid gradient update
        frozen_actor_params = actor_state.params

        def loss_fn(critic_params: FrozenDict,
                    rng: Any,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    discount: jnp.ndarray):
            """compute loss for a single transition"""
            rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

            # Repeat observations
            repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0),
                                             repeats=self.num_random, axis=0)
            repeat_next_observations = jnp.repeat(jnp.expand_dims(next_observation, axis=0),
                                                  repeats=self.num_random, axis=0)

            # Critic loss
            q1, q2 = self.critic.apply({"params": critic_params}, observation, action)

            # Target Q value
            if self.max_target_backup:
                _, next_actions, logp_next_actions = self.actor.apply({"params": frozen_actor_params},
                                                                      rng1, repeat_next_observations)
                next_q1s, next_q2s = self.critic.apply({"params": critic_target_params}, 
                                                       repeat_next_observations, next_actions)
                next_qs = jnp.minimum(next_q1s, next_q2s)
                max_target_idx = jnp.argmax(next_qs, keepdims=True)
                logp_next_action = jnp.take_along_axis(logp_next_actions, max_target_idx, 0).squeeze()
                next_q = jnp.take_along_axis(next_qs, max_target_idx, 0).squeeze()
            else:
                _, next_action, logp_next_action = self.actor.apply({"params": frozen_actor_params},
                                                                    rng1, next_observation)
                next_q1, next_q2 = self.critic.apply({"params": critic_target_params},
                                                     next_observation, next_action)
                next_q = jnp.minimum(next_q1, next_q2)

            if self.backup_entropy:
                next_q -= sac_alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q
            critic_loss1 = (q1 - target_q)**2
            critic_loss2 = (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2

            # CQL loss
            cql_random_actions = jax.random.uniform(key=rng2,
                                                    shape=(self.num_random, self.act_dim),
                                                    minval=-self.max_action,
                                                    maxval=self.max_action)
            _, cql_sampled_actions, cql_logp = self.actor.apply({"params": frozen_actor_params},
                                                                rng3, repeat_observations)
            _, cql_next_actions, cql_logp_next_action = self.actor.apply({"params": frozen_actor_params},
                                                                         rng4, repeat_next_observations)

            cql_random_q1, cql_random_q2 = self.critic.apply({"params": critic_params},
                                                             repeat_observations,
                                                             cql_random_actions)
            cql_q1, cql_q2 = self.critic.apply({"params": critic_params},
                                               repeat_observations,
                                               cql_sampled_actions)
            cql_next_q1, cql_next_q2 = self.critic.apply({"params": critic_params},
                                                         repeat_observations,
                                                         cql_next_actions)

            # Simulate logsumexp() for continuous actions
            random_density = jnp.log(0.5**self.act_dim)
            cql_concat_q1 = jnp.concatenate([cql_random_q1-random_density,
                                             cql_next_q1-cql_logp_next_action,
                                             cql_q1-cql_logp])
            cql_concat_q2 = jnp.concatenate([cql_random_q2-random_density,
                                             cql_next_q2-cql_logp_next_action,
                                             cql_q2-cql_logp])

            # CQL0: conservative penalty ==> dominate by the max(cql_concat_q)
            ood_q1 = jax.scipy.special.logsumexp(cql_concat_q1)
            ood_q2 = jax.scipy.special.logsumexp(cql_concat_q2)

            cql_diff1 = jnp.clip(ood_q1 - q1, self.cql_clip_diff_min, self.cql_clip_diff_max)
            cql_diff2 = jnp.clip(ood_q2 - q2, self.cql_clip_diff_min, self.cql_clip_diff_max)

            if self.with_lagrange:
                cql_loss1 = cql_alpha * (cql_diff1 - self.lagrange_thresh) * self.min_q_weight
                cql_loss2 = cql_alpha * (cql_diff2 - self.lagrange_thresh) * self.min_q_weight
            else:
                cql_loss1 = cql_diff1 * self.min_q_weight
                cql_loss2 = cql_diff2 * self.min_q_weight

            total_loss = critic_loss + cql_loss1 + cql_loss2
            log_info = {
                "critic_loss": critic_loss,
                "cql_loss1": cql_loss1,
                "cql_loss2": cql_loss2,
                "q1": q1,
                "q2": q2,
                "target_q": target_q,
                "ood_q1": ood_q1,
                "ood_q2": ood_q2,
                "cql_q1": cql_q1.mean(),
                "cql_q2": cql_q2.mean(),
                "random_q1": cql_random_q1.mean(),
                "random_q2": cql_random_q2.mean(),
                "min_q_weight": self.min_q_weight,
                "logp_next_action": logp_next_action,
                "cql_diff1": cql_diff1,
                "cql_diff2": cql_diff2,
                "cql_alpha": cql_alpha
            }

            return total_loss, log_info

        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, argnums=(0), has_aux=True),
                           in_axes=(None, 0, 0, 0, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))
        (_, log_info), grads = grad_fn(critic_state.params,
                                       keys, 
                                       batch.observations,
                                       batch.actions,
                                       batch.rewards,
                                       batch.next_observations,
                                       batch.discounts)

        extra_log_info = {
            "q1_max": log_info["q1"].max(),
            "q1_min": log_info["q1"].min(),
            "target_q_max": log_info["target_q"].max(),
            "target_q_min": log_info["target_q"].min(),
            "critic_loss_max": log_info["critic_loss"].max(),
            "critic_loss_min": log_info["critic_loss"].min(),
        }

        grads = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), grads)
        log_info = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), log_info)

        new_critic_state = critic_state.apply_gradients(grads=grads)
        new_critic_target_params = target_update(new_critic_state.params, critic_target_params, self.tau)
        return new_critic_state, new_critic_target_params, {**log_info, **extra_log_info}

    @functools.partial(jax.jit, static_argnames=("self", "bc"))
    def train_step(self, batch, actor_key, critic_key, alpha_state, actor_state,
                   critic_state, critic_target_params, cql_alpha, bc):
        new_alpha_state, new_actor_state, actor_log_info = self.actor_alpha_train_step(
            batch, actor_key, alpha_state, actor_state, critic_state, bc)
        sac_alpha = actor_log_info["alpha"]

        new_critic_state, new_critic_target_params, critic_log_info = self.critic_train_step(
            batch, critic_key, actor_state, critic_state, critic_target_params, sac_alpha, cql_alpha)

        return new_alpha_state, new_actor_state, new_critic_state, new_critic_target_params, {
            **actor_log_info, **critic_log_info}

    @functools.partial(jax.jit, static_argnames=("self"))
    def lagrange_train_step(self,
                            cql_alpha_state: train_state.TrainState,
                            cql_diff1: float,
                            cql_diff2: float):
        def loss_fn(params):
            log_cql_alpha = self.log_cql_alpha.apply({"params": params})
            cql_alpha = jnp.clip(jnp.exp(log_cql_alpha), a_min=0.0, a_max=1000000.0)
            cql_alpha_loss1 = cql_alpha * (cql_diff1 - self.lagrange_thresh) * self.min_q_weight
            cql_alpha_loss2 = cql_alpha * (cql_diff2 - self.lagrange_thresh) * self.min_q_weight
            cql_alpha_loss = (-cql_alpha_loss1-cql_alpha_loss2) * 0.5
            return cql_alpha_loss, {"cql_alpha_loss": cql_alpha_loss}
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grads = grad_fn(cql_alpha_state.params)
        new_cql_alpha_state = cql_alpha_state.apply_gradients(grads=grads)
        return new_cql_alpha_state, log_info

    def update(self, batch: Batch):
        self.update_step += 1
        bc = self.update_step <= self.bc_timesteps
        log_cql_alpha = self.log_cql_alpha.apply({"params": self.cql_alpha_state.params}) if self.with_lagrange else 0.0
        cql_alpha = jnp.clip(jnp.exp(log_cql_alpha), a_min=0.0, a_max=1000000.0)
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)

        (self.alpha_state,
         self.actor_state,
         self.critic_state,
         self.critic_target_params,
         log_info) = self.train_step(batch,
                                     actor_key,
                                     critic_key,
                                     self.alpha_state,
                                     self.actor_state,
                                     self.critic_state,
                                     self.critic_target_params,
                                     cql_alpha,
                                     bc)

        if self.with_lagrange:
            self.cql_alpha_state, cql_log_info = self.lagrange_train_step(self.cql_alpha_state,
                                                                          log_info["cql_diff1"],
                                                                          log_info["cql_diff2"])
            log_info.update(cql_log_info)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)

    def load(self, ckpt_dir, step):
        self.actor_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.actor_state,
                                                          step=step, prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=self.critic_state,
                                                          step=step, prefix="critic_")
