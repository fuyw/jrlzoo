from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict
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
            tx=optax.adam(learning_rate=lr_actor)
        )

        # CQL parameters
        self.num_random = num_random
        self.with_lagrange = with_lagrange
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.min_q_weight = 1.0 if with_lagrange else min_q_weight
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

    @functools.partial(jax.jit, static_argnames=("self"))
    def lagrange_train_step(self,
                            cql_alpha_state: train_state.TrainState,
                            cql_diff1: float,
                            cql_diff2: float):
        def loss_fn(params):
            log_cql_alpha = self.log_cql_alpha.apply({"params": params})
            cql_alpha = jnp.clip(jnp.exp(log_cql_alpha), a_min=0.0, a_max=1000000.0)
            cql_alpha_loss1 = cql_alpha * (cql_diff1 - self.lagrange_thresh)
            cql_alpha_loss2 = cql_alpha * (cql_diff2 - self.lagrange_thresh)
            cql_alpha_loss = (-cql_alpha_loss1-cql_alpha_loss2) * 0.5
            return cql_alpha_loss, {"cql_alpha_loss": cql_alpha_loss}
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grads = grad_fn(cql_alpha_state.params)
        new_cql_alpha_state = cql_alpha_state.apply_gradients(grads=grads)
        return new_cql_alpha_state, log_info

    @functools.partial(jax.jit, static_argnames=("self", "bc"))
    def cql_train_step(self,
                       batch: Batch,
                       key: Any,
                       alpha_state: train_state.TrainState,
                       actor_state: train_state.TrainState,
                       critic_state: train_state.TrainState,
                       critic_target_params: FrozenDict,
                       cql_alpha: float = 1.0,
                       bc: bool = False):

        # prepare prozen params to avoid gradient update
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        def loss_fn(alpha_params: FrozenDict,
                    actor_params: FrozenDict,
                    critic_params: FrozenDict,
                    rng: Any,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    discount: jnp.ndarray):
            """compute loss for a single transition"""
            rng, rng1, rng2 = jax.random.split(rng, 3)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply({"params": actor_params}, rng1, observation)

            # Alpha loss: stop gradient to avoid affecting Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha) 

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.minimum(sampled_q1, sampled_q2)

            # Actor loss
            if bc:
                logp_bc = self.actor.apply({"params": actor_params}, observation, action, method=Actor.get_logp)
                actor_loss = alpha * logp - logp_bc  # minimize KL-divergence w.r.t. behavior policy
            else:
                actor_loss = alpha * logp - sampled_q

            # Repeat observations
            repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0), repeats=self.num_random, axis=0)
            repeat_next_observations = jnp.repeat(jnp.expand_dims(next_observation, axis=0), repeats=self.num_random, axis=0)

            # Critic loss
            q1, q2 = self.critic.apply({"params": critic_params}, observation, action)

            # Target Q value
            if self.max_target_backup:
                _, next_actions, logp_next_actions = self.actor.apply({"params": frozen_actor_params}, rng2, repeat_next_observations)
                next_q1s, next_q2s = self.critic.apply({"params": critic_target_params}, repeat_next_observations, next_actions)
                next_qs = jnp.minimum(next_q1s, next_q2s)
                max_target_idx = jnp.argmax(next_qs, keepdims=True)
                logp_next_action = jnp.take_along_axis(logp_next_actions, max_target_idx, 0).squeeze()
                next_q = jnp.take_along_axis(next_qs, max_target_idx, 0).squeeze()
            else:
                _, next_action, logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng2, next_observation)
                next_q1, next_q2 = self.critic.apply({"params": critic_target_params}, next_observation, next_action)
                next_q = jnp.minimum(next_q1, next_q2)

            if self.backup_entropy:
                next_q -= alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q
            critic_loss1 = 0.5 * (q1 - target_q)**2
            critic_loss2 = 0.5 * (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2

            # CQL loss
            rng3, rng4 = jax.random.split(rng, 2)
            cql_random_actions = jax.random.uniform(
                rng3, shape=(self.num_random, self.act_dim), minval=-self.max_action, maxval=self.max_action)

            _, cql_sampled_actions, cql_logp = self.actor.apply({"params": frozen_actor_params}, rng3, repeat_observations)
            _, cql_next_actions, cql_logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng4, repeat_next_observations)

            cql_random_q1, cql_random_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_random_actions)
            cql_q1, cql_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_sampled_actions)
            cql_next_q1, cql_next_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_next_actions)

            # Simulate logsumexp() for continuous actions
            random_density = jnp.log(0.5**self.act_dim)
            cql_concat_q1 = jnp.concatenate([cql_random_q1-random_density, cql_next_q1-cql_logp_next_action, cql_q1-cql_logp])
            cql_concat_q2 = jnp.concatenate([cql_random_q2-random_density, cql_next_q2-cql_logp_next_action, cql_q2-cql_logp])

            # CQL0: conservative penalty ==> dominate by the max(cql_concat_q)
            ood_q1 = jax.scipy.special.logsumexp(cql_concat_q1)
            ood_q2 = jax.scipy.special.logsumexp(cql_concat_q2)

            cql_diff1 = jnp.clip(ood_q1 - q1, self.cql_clip_diff_min, self.cql_clip_diff_max)
            cql_diff2 = jnp.clip(ood_q2 - q2, self.cql_clip_diff_min, self.cql_clip_diff_max)
            cql_loss1 = cql_alpha * cql_diff1 * self.min_q_weight
            cql_loss2 = cql_alpha * cql_diff2 * self.min_q_weight

            # Loss weight form Dopamine
            total_loss = critic_loss + actor_loss + alpha_loss + cql_loss1 + cql_loss2
            log_info = {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "cql_loss1": cql_loss1,
                "cql_loss2": cql_loss2,
                "q1": q1,
                "q2": q2,
                "target_q": target_q,
                "sampled_q": sampled_q,
                "ood_q1": ood_q1,
                "ood_q2": ood_q2,
                "cql_q1": cql_q1.mean(),
                "cql_q2": cql_q2.mean(),
                "random_q1": cql_random_q1.mean(),
                "random_q2": cql_random_q2.mean(),
                "alpha": alpha,
                "logp": logp,
                "min_q_weight": self.min_q_weight,
                "logp_next_action": logp_next_action,
                "cql_diff1": cql_diff1,
                "cql_diff2": cql_diff2,
                "cql_alpha": cql_alpha
            }

            return total_loss, log_info

        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
                           in_axes=(None, None, None, 0, 0, 0, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))
        (_, log_info), grads = grad_fn(alpha_state.params,
                                       actor_state.params,
                                       critic_state.params,
                                       keys, 
                                       batch.observations,
                                       batch.actions,
                                       batch.rewards,
                                       batch.next_observations,
                                       batch.discounts)
        grads = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), grads)
        log_info = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        alpha_grads, actor_grads, critic_grads = grads

        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)
        new_critic_target_params = target_update(new_critic_state.params, critic_target_params, self.tau)
        return new_alpha_state, new_actor_state, new_critic_state, new_critic_target_params, log_info

    def update(self, batch: Batch):
        self.update_step += 1
        bc = self.update_step <= self.bc_timesteps
        log_cql_alpha = self.log_cql_alpha.apply({"params": self.cql_alpha_state.params}) if self.with_lagrange else 0.0
        cql_alpha = jnp.clip(jnp.exp(log_cql_alpha), a_min=0.0, a_max=1000000.0)
        self.rng, key = jax.random.split(self.rng)
        self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params, log_info = self.cql_train_step(
            batch, key, self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params, cql_alpha, bc)
        if self.with_lagrange:
            self.cql_alpha_state, cql_log_info = self.lagrange_train_step(self.cql_alpha_state, log_info["cql_diff1"], log_info["cql_diff2"])
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


##################
# Ensemble Agent #
##################
class EnsembleDense(nn.Module):
    ensemble_num: int
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.array) -> jnp.array:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param("kernel", self.kernel_init,
                            (self.ensemble_num, inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.einsum("ij,ijk->ik", inputs, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.ensemble_num, self.features))
            bias = jnp.asarray(bias, self.dtype)
            y += bias
        return y


class EnsembleCritic(nn.Module):
    ensemble_num: int = 3
    hid_dim: int = 256
    initializer: str = "glorot_uniform"

    def setup(self):
        self.l1 = EnsembleDense(self.ensemble_num,
                                features=self.hid_dim,
                                kernel_init=init_fn(self.initializer, 1e-2),
                                name="fc1")
        self.l2 = EnsembleDense(self.ensemble_num,
                                features=self.hid_dim,
                                kernel_init=init_fn(self.initializer, 1e-2),
                                name="fc2")
        self.out_layer = EnsembleDense(self.ensemble_num,
                                       features=1,
                                       kernel_init=init_fn(self.initializer, 1e-2),
                                       name="out")

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)  # (E, obs_dim+act_dim)
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        q = self.out_layer(x)  # (E, 1)
        return q.squeeze(-1)   # (E,)


class EnsembleDoubleCritic(nn.Module):
    ensemble_num: int = 3
    hid_dim: int = 256
    initializer: str = "glorot_uniform"

    def setup(self):
        self.critic1 = EnsembleCritic(self.ensemble_num, self.hid_dim, self.initializer)
        self.critic2 = EnsembleCritic(self.ensemble_num, self.hid_dim, self.initializer)
    
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2


class EnsembleActor(nn.Module):
    act_dim: int
    ensemble_num: int = 3
    max_action: float = 1.0
    hid_dim: int = 256
    initializer: str = "glorot_uniform"

    def setup(self):
        self.l1 = EnsembleDense(self.ensemble_num,
                                features=self.hid_dim,
                                kernel_init=init_fn(self.initializer, 1e-2),
                                name="fc1")
        self.l2 = EnsembleDense(self.ensemble_num,
                                features=self.hid_dim,
                                kernel_init=init_fn(self.initializer, 1e-2),
                                name="fc2")
        self.mu_layer = EnsembleDense(self.ensemble_num,
                                      features=self.act_dim,
                                      kernel_init=init_fn(self.initializer, 1e-2))
        self.std_layer = EnsembleDense(self.ensemble_num,
                                       features=self.act_dim,
                                       kernel_init=init_fn(self.initializer, 1e-2))

    def __call__(self, rng: Any, observation: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(self.l1(observation))  # (E, obs_dim) ==> (E, hid_dim)
        x = nn.relu(self.l2(x))
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = jnp.exp(log_std)  # (E, act_dim)
        mu = self.mu_layer(x)   # (E, act_dim)

        # ensemble avg
        avg_mu = mu.mean(0)     # (act_dim,)
        avg_var = (mu**2 + std**2).mean(0) - avg_mu**2  # (act_dim,)
        avg_std = jnp.sqrt(avg_var)
        avg_std = jnp.clip(avg_std, jnp.exp(LOG_STD_MIN), jnp.exp(LOG_STD_MAX))

        mean_action = nn.tanh(avg_mu)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(avg_mu, avg_std),
            distrax.Block(distrax.Tanh(), ndims=1))
        sampled_action, logp = action_distribution.sample_and_log_prob(seed=rng)  # (act_dim), ()
        return mean_action*self.max_action, sampled_action*self.max_action, logp

    def get_logp(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(self.l1(observation))  # (E, obs_dim) ==> (E, hid_dim)
        x = nn.relu(self.l2(x))
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = jnp.exp(log_std)  # (E, act_dim)
        mu = self.mu_layer(x)   # (E, act_dim)

        # ensemble avg
        avg_mu = mu.mean(0)     # (act_dim,)
        avg_var = (mu**2 + std**2).mean(0) - avg_mu**2  # (act_dim,)
        avg_std = jnp.sqrt(avg_var)
        avg_std = jnp.clip(avg_std, jnp.exp(LOG_STD_MIN), jnp.exp(LOG_STD_MAX))

        action_distribution = distrax.Normal(avg_mu, avg_std)
        raw_action = atanh(action)
        logp = action_distribution.log_prob(raw_action).sum(-1)
        logp -= 2*(jnp.log(2) - raw_action - jax.nn.softplus(-2*raw_action)).sum(-1)
        return logp

    def get_actions(self, observation: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(self.l1(observation))  # (E, obs_dim) ==> (E, hid_dim)
        x = nn.relu(self.l2(x))
        mu = self.mu_layer(x)   # (E, act_dim)
        mean_action = nn.tanh(mu)
        return mean_action*self.max_action


class Off2OnAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 ensemble_num: int = 3,
                 max_action: float = 1.0,
                 hidden_dim: int = 256,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 target_entropy: float = None,
                 initializer: str = "glorot_uniform"):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ensemble_num = ensemble_num

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.max_action = max_action
        if target_entropy is None:
            self.target_entropy = -act_dim / 2  # dopamine setting
        else:
            self.target_entropy = target_entropy

        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key, weight_key = jax.random.split(self.rng, 4)

        # Dummy inputs
        dummy_obs = jnp.ones([ensemble_num, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([ensemble_num, act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = EnsembleActor(act_dim, ensemble_num)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=lr)
        )

        # Initialize the Critic
        self.critic = EnsembleDoubleCritic(ensemble_num)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=lr)
        )

        # Entropy tuning
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.adam(learning_rate=lr)
        )

        # Weight network
        self.weight_net = Critic()
        weight_params = self.weight_net.init(weight_key, dummy_obs, dummy_act)["params"]
        self.weight_state = train_state.TrainState.create(
            apply_fn=self.weight.apply,
            params=weight_params,
            tx=optax.adam(learning_rate=lr))

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   online_batch: Batch,
                   offline_batch: Batch,
                   key: Any,
                   weight_state: train_state.TrainState,
                   alpha_state: train_state.TrainState,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        observations = jnp.repeat(jnp.expand_dims(batch.observations, axis=0),
                                  repeats=self.ensemble_num, axis=0)        
        actions = jnp.repeat(jnp.expand_dims(batch.actions, axis=0),
                             repeats=self.ensemble_num, axis=0)
        next_observations = jnp.repeat(jnp.expand_dims(batch.next_observations, axis=0),
                                       repeats=self.ensemble_num, axis=0)

        def loss_fn(alpha_params: FrozenDict,
                    actor_params: FrozenDict,
                    critic_params: FrozenDict,
                    weight_params: FrozenDict,
                    rng: Any,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    discount: jnp.ndarray,
                    online_observation: jnp.ndarray,
                    online_action: jnp.ndarray,
                    offline_observation: jnp.ndarray,
                    offline_action: jnp.ndarray):
            rng1, rng2 = jax.random.split(rng, 2)

            # weight loss: maximize lower bound of JS divergence
            offline_weight = nn.relu(self.weight_net.apply(
                {"params": weight_params}, offline_observation, offline_action))
            offline_f_star = -nn.log(2.0 / (offline_weight + 1) + 1e-6)
            online_weight = nn.relu(self.weight_net.apply(
                {"params": weight_params}, online_observation, online_action))
            online_f_prime = nn.log(2.0 * online_weight / (online_weight + 1) + 1e-6)
            weight_loss = offline_f_star - online_f_prime

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply(
                {"params": actor_params}, rng1, observation)  # logp.sahpe = ()
            sampled_action = jnp.repeat(jnp.expand_dims(sampled_action, axis=0),
                                        repeats=self.ensemble_num, axis=0)

            # Alpha loss: stop gradient to avoid affecting Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(
                logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha)  # alpha.shape = ()

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply(
                {"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.minimum(sampled_q1, sampled_q2).mean(0)  # sampled_q.shape = ()

            # Actor loss
            actor_loss = alpha * logp - sampled_q

            q1, q2 = self.critic.apply(
                {"params": critic_params}, observation, action)  # (E,), (E,)
            _, next_action, logp_next_action = self.actor.apply(
                {"params": frozen_actor_params}, rng2, next_observation)
            next_action = jnp.repeat(jnp.expand_dims(next_action, axis=0),
                                     repeats=self.ensemble_num, axis=0)
            next_q1, next_q2 = self.critic.apply(
                {"params": critic_target_params}, next_observation, next_action)  # (E,)
            next_q = jnp.minimum(next_q1, next_q2).mean(0) - alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q  # ()
            critic_loss1 = ((q1 - target_q)**2).sum(0)
            critic_loss2 = ((q2 - target_q)**2).sum(0)
            critic_loss = critic_loss1 + critic_loss2

            total_loss = critic_loss + actor_loss + alpha_loss + weight_loss
            log_info = {
                "critic_loss1": critic_loss1,
                "critic_loss2": critic_loss2,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "weight_loss": weight_loss,
                "offline_f_star": offline_f_star,
                "online_f_prime": online_f_prime,
                "q1": q1.mean(),
                "q2": q2.mean(),
                "target_q": target_q,
                "alpha": alpha,
                "logp": logp
            }
            return total_loss, log_info

        grad_fn = jax.vmap(jax.value_and_grad(loss_fn,
                                              argnums=(0, 1, 2, 3),
                                              has_aux=True),
                           in_axes=(None, None, None, None, 0,
                                    1, 1, 0, 1, 0, 0, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))

        (_, log_info), grads = grad_fn(alpha_state.params,
                                       actor_state.params,
                                       critic_state.params,
                                       weight_state.params,
                                       keys,
                                       observations,
                                       actions,
                                       batch.rewards,
                                       next_observations,
                                       batch.discounts)
        grads = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), grads)


        
        log_info = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), log_info)

        alpha_grads, actor_grads, critic_grads, weight_grads = grads
        new_weight_state = weight_state.apply_gradients(grads=weight_grads)
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)
        new_critic_target_params = target_update(new_critic_state.params,
                                                 critic_target_params,
                                                 self.tau)
        
        new_weights = weight_state.apply({"params": weight_state.params},
                                         batch.observations,
                                         batch.actions)
        new_weights = nn.relu(new_weights)
        normalized_weights = (new_weights ** (1/5.0)) / ()

        return new_weight_state, new_alpha_state, new_actor_state, new_critic_state, new_critic_target_params, log_info

    # Updating the priority buffer
    # """
    # with torch.no_grad():
    #     weight = self.w_activation(self.weight_net(obs, actions))

    #     normalized_weight = (weight ** (1 / self.temperature)) / (
    #         (offline_weight ** (1 / self.temperature)).mean() + 1e-10
    #     )

    # new_priority = normalized_weight.clamp(0.001, 1000)

    def update(self, batch: Batch):
        self.rng, key = jax.random.split(self.rng, 2)
        (self.weight_state, self.alpha_state,
         self.actor_state, self.critic_state,
         self.critic_target_params, log_info) = self.train_step(
            batch, key, self.alpha_state, self.actor_state,
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

    def load_cql_ckpts(self, cql_agents):
        def load_actor_params(actor_params, cql_actor_params):
            for idx, params in enumerate(cql_actor_params):
                actor_params["fc1"]["kernel"] = actor_params["fc1"]["kernel"].at[idx].set(params["net"]["Dense_0"]["kernel"])
                actor_params["fc1"]["bias"] = actor_params["fc1"]["bias"].at[idx].set(params["net"]["Dense_0"]["bias"])
                actor_params["fc2"]["kernel"] = actor_params["fc2"]["kernel"].at[idx].set(params["net"]["Dense_1"]["kernel"])
                actor_params["fc2"]["bias"] = actor_params["fc2"]["bias"].at[idx].set(params["net"]["Dense_1"]["bias"])
                actor_params["mu_layer"]["kernel"] = actor_params["mu_layer"]["kernel"].at[idx].set(params["mu_layer"]["kernel"])
                actor_params["mu_layer"]["bias"] = actor_params["mu_layer"]["bias"].at[idx].set(params["mu_layer"]["bias"])
                actor_params["std_layer"]["kernel"] = actor_params["std_layer"]["kernel"].at[idx].set(params["std_layer"]["kernel"])
                actor_params["std_layer"]["bias"] = actor_params["std_layer"]["bias"].at[idx].set(params["std_layer"]["bias"])
            return frozen_dict.freeze(actor_params)

        def load_critic_params(critic_params, cql_critic_params):
            for idx, params in enumerate(cql_critic_params):
                critic_params["critic1"]["fc1"]["kernel"] = critic_params["critic1"]["fc1"]["kernel"].at[idx].set(params["VmapCritic_0"]["net"]["Dense_0"]["kernel"][0])
                critic_params["critic1"]["fc1"]["bias"] = critic_params["critic1"]["fc1"]["bias"].at[idx].set(params["VmapCritic_0"]["net"]["Dense_0"]["bias"][0])
                critic_params["critic1"]["fc2"]["kernel"] = critic_params["critic1"]["fc2"]["kernel"].at[idx].set(params["VmapCritic_0"]["net"]["Dense_1"]["kernel"][0])
                critic_params["critic1"]["fc2"]["bias"] = critic_params["critic1"]["fc2"]["bias"].at[idx].set(params["VmapCritic_0"]["net"]["Dense_1"]["bias"][0])
                critic_params["critic1"]["out"]["kernel"] = critic_params["critic1"]["out"]["kernel"].at[idx].set(params["VmapCritic_0"]["out_layer"]["kernel"][0])
                critic_params["critic1"]["out"]["bias"] = critic_params["critic1"]["out"]["bias"].at[idx].set(params["VmapCritic_0"]["out_layer"]["bias"][0])

                critic_params["critic2"]["fc1"]["kernel"] = critic_params["critic2"]["fc1"]["kernel"].at[idx].set(params["VmapCritic_0"]["net"]["Dense_0"]["kernel"][1])
                critic_params["critic2"]["fc1"]["bias"] = critic_params["critic2"]["fc1"]["bias"].at[idx].set(params["VmapCritic_0"]["net"]["Dense_0"]["bias"][1])
                critic_params["critic2"]["fc2"]["kernel"] = critic_params["critic2"]["fc2"]["kernel"].at[idx].set(params["VmapCritic_0"]["net"]["Dense_1"]["kernel"][1])
                critic_params["critic2"]["fc2"]["bias"] = critic_params["critic2"]["fc2"]["bias"].at[idx].set(params["VmapCritic_0"]["net"]["Dense_1"]["bias"][1])
                critic_params["critic2"]["out"]["kernel"] = critic_params["critic2"]["out"]["kernel"].at[idx].set(params["VmapCritic_0"]["out_layer"]["kernel"][1])
                critic_params["critic2"]["out"]["bias"] = critic_params["critic2"]["out"]["bias"].at[idx].set(params["VmapCritic_0"]["out_layer"]["bias"][1])
            return frozen_dict.freeze(critic_params)

        cql_actor_params = [agent.actor_state.params.unfreeze() for agent in cql_agents]
        cql_critic_params = [agent.critic_state.params.unfreeze() for agent in cql_agents]

        actor_params = load_actor_params(self.actor_state.params.unfreeze(),
                                         cql_actor_params)
        critic_params = load_critic_params(self.critic_state.params.unfreeze(), 
                                           cql_critic_params)

        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=frozen_dict.freeze(actor_params),
            tx=optax.adam(self.lr)
        )

        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=frozen_dict.freeze(critic_params),
            tx=optax.adam(self.lr)
        )

    def get_actions(self, observation): 
        @jax.jit
        def _get_actions(params, observation):
            observation = jnp.repeat(jnp.expand_dims(observation, axis=0), repeats=self.ensemble_num, axis=0)
            actions = self.actor.apply({"params": params}, observation, method=EnsembleActor.get_actions)
            return actions
        actions = _get_actions(self.actor_state.params, observation)
        return np.asarray(actions)
