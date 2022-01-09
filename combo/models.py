from typing import Any, Callable, Optional
import functools
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from utils import Batch


LOG_STD_MAX = 2.
LOG_STD_MIN = -20.

kernel_initializer = jax.nn.initializers.glorot_uniform()


class Actor(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, rng: Any, observation: jnp.ndarray):
        x = nn.relu(nn.Dense(256, kernel_init=kernel_initializer, name="fc1")(observation))
        x = nn.relu(nn.Dense(256, kernel_init=kernel_initializer, name="fc2")(x))
        x = nn.Dense(2 * self.act_dim, kernel_init=kernel_initializer, name="fc3")(x)

        mu, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
 
        mean_action = jnp.tanh(mu)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1)
        )
        sampled_action, logp = action_distribution.sample_and_log_prob(seed=rng)
        return mean_action, sampled_action, logp


class Critic(nn.Module):
    hid_dim: int = 256

    @nn.compact
    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observation, action], axis=-1)
        q = nn.relu(nn.Dense(self.hid_dim, kernel_init=kernel_initializer, name="fc1")(x))
        q = nn.relu(nn.Dense(self.hid_dim, kernel_init=kernel_initializer, name="fc2")(q))
        q = nn.Dense(1, kernel_init=kernel_initializer, name="fc3")(q)
        return q


class DoubleCritic(nn.Module):
    hid_dim: int = 256

    def setup(self):
        self.critic1 = Critic(self.hid_dim)
        self.critic2 = Critic(self.hid_dim)

    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observation, action)
        q2 = self.critic2(observation, action)
        return q1, q2


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)
    
    def __call__(self):
        return self.value


class EnsembleDense(nn.Module):
    num_members: int
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
                            (self.num_members, inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.einsum("ij,ijk->ik", inputs, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.num_members, self.features))
            bias = jnp.asarray(bias, self.dtype)
            y += bias
        return y


class GaussianMLP(nn.Module):
    num_members: int
    out_dim: int
    hid_dim: int = 256
    max_log_var: float = 0.5
    min_log_var: float = -10.0

    def setup(self):
        self.l1 = EnsembleDense(num_members=self.num_members, features=self.hid_dim, name="fc1")
        self.l2 = EnsembleDense(num_members=self.num_members, features=self.hid_dim, name="fc2")
        self.l3 = EnsembleDense(num_members=self.num_members, features=self.hid_dim, name="fc3")
        self.l4 = EnsembleDense(num_members=self.num_members, features=self.hid_dim, name="fc4")
        self.l5 = EnsembleDense(num_members=self.num_members, features=self.out_dim*2, name="fc5")

    def __call__(self, x):
        x = nn.swish(self.l1(x))
        x = nn.swish(self.l2(x))
        x = nn.swish(self.l3(x))
        x = nn.swish(self.l4(x))
        x = self.l5(x)

        mu, log_var = jnp.split(x, 2, axis=-1)
        # TODO:
        # log_var = jnp.clip(log_std, self.min_log_var, self.max_log_var)
        log_var = self.max_log_var - jax.nn.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + jax.nn.softplus(log_var - self.min_log_var)
        return mu, log_var


class COMBOAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 lr_actor: float = 3e-4,
                 auto_entropy_tuning: bool = True,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = False,
                 num_random: int = 10,
                 with_lagrange: bool = False,
                 lagrange_thresh: int = 5.0,

                 # COMBO
                 lr_model: float = 1e-3,
                 weight_decay: float = 3e-5,
                 num_members: int = 7,
                 max_patience: int = 5,
                 real_ratio: float = 0.1,
                 holdout_ratio: float = 0.1):

        self.update_step = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.lr_actor = lr_actor
        self.auto_entropy_tuning = auto_entropy_tuning
        self.backup_entropy = backup_entropy
        if target_entropy is None:
            self.target_entropy = -self.act_dim
        else:
            self.target_entropy = target_entropy

        # COMBO parameters
        self.lr_model = lr_model
        self.weight_decay = weight_decay
        self.max_patience = max_patience
        self.real_ratio = real_ratio
        self.holdout_ratio = holdout_ratio
        self.num_members = num_members

        # Initialize random keys
        self.rng = jax.random.PRNGKey(seed)
        self.rng, actor_key, critic_key, model_key = jax.random.split(self.rng, 4)

        # Dummy inputs
        dummy_obs = jnp.ones([1, self.obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, self.act_dim], dtype=jnp.float32)
        dummy_model_inputs = jnp.ones([self.num_members, self.obs_dim+self.act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(self.act_dim)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.adam(self.lr_actor))

        # Initialize the Critic
        self.critic = DoubleCritic()
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=Critic.apply,
            params=critic_params,
            tx=optax.adam(self.lr))

        # Initialize the Dynamics Model
        self.model = GaussianMLP(self.num_members, self.obs_dim+1)
        model_params = self.model.init(model_key, dummy_model_inputs)["params"]
        self.model_state = train_state.TrainState.create(
            apply_fn=GaussianMLP.apply,
            params=model_params,
            tx=optax.adamw(learning_rate=self.lr_model, weight_decay=self.weight_decay))

        # Entropy tuning
        if self.auto_entropy_tuning:
            self.rng, alpha_key = jax.random.split(self.rng, 2)
            self.log_alpha = Scalar(0.0)
            self.alpha_state = train_state.TrainState.create(
                apply_fn=None,
                params=self.log_alpha.init(alpha_key)["params"],
                tx=optax.adam(self.lr)
            )

        # CQL parameters
        self.num_random = num_random
        self.with_lagrange = with_lagrange
        self.min_q_weight = 5.0 if not with_lagrange else 1.0
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.rng, cql_key = jax.random.split(self.rng, 2)
            self.log_cql_alpha = Scalar(0.0)  # 1.0
            self.cql_alpha_state = train_state.TrainState.create(
                apply_fn=None,
                params=self.log_cql_alpha.init(cql_key)["params"],
                tx=optax.adam(self.lr))
 
    def train_model(self, replay_buffer, batch_size=1024, epochs=100):
        @jax.jit
        def train_loss_fn(params, x, y):
            mu, log_var = self.model.apply({'params': params}, x)  # (7, 12)
            inv_var = jnp.exp(-log_var)
            mse_loss = jnp.mean(jnp.mean(jnp.square(mu - y) * inv_var, axis=-1), axis=-1)
            var_loss = jnp.mean(jnp.mean(log_var, axis=-1), axis=-1)
            total_loss = mse_loss + var_loss
            return total_loss, {'mse_loss': mse_loss, 'var_loss': var_loss, 'train_loss': total_loss}

        @jax.jit
        def val_loss_fn(params, x, y):
            mu, log_var = jax.lax.stop_gradient(self.model.apply({'params': params}, x))
            inv_var = jnp.exp(-log_var)
            mse_loss = jnp.mean(jnp.mean(jnp.square(mu - y) * inv_var, axis=-1), axis=-1)
            return mse_loss

        grad_fn = jax.vmap(jax.value_and_grad(train_loss_fn, has_aux=True), in_axes=(None, 1, 1))

        # Preparing inputs and outputs
        observations = replay_buffer.observations
        actions = replay_buffer.actions
        next_observations = replay_buffer.next_observations
        rewards = replay_buffer.rewards.reshape(-1, 1)
        delta_observations = next_observations - observations
        inputs = np.concatenate([observations, actions], axis=-1)
        targets = np.concatenate([rewards, delta_observations], axis=-1)

        # Split into training and holdout sets
        num_holdout = int(inputs.shape[0] * self.holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_members, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_members, 1, 1])

        batch_num = int(np.ceil(inputs.shape[0] / batch_size))

        for epoch in trange(epochs, desc='[Training the dynamics model]'):
            shuffled_idxs = np.concatenate([np.random.permutation(np.arange(inputs.shape[0])).reshape(1, -1)
                                            for _ in range(self.num_members)], axis=0)  # (7, N)
            for i in range(batch_num):
                batch_idxs = shuffled_idxs[:, i*batch_size:(i+1)*batch_size]
                batch_inputs = inputs[batch_idxs]
                batch_targets = targets[batch_idxs]


            # run validation

            # print log info
            print(f'Epoch {epoch+1}: ')

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_model_step(self, model_state, batch_inputs, batch_targets):

        def loss_fn(params: FrozenDict,
                    x: jnp.ndarray,
                    y: jnp.ndarray):
            mu, log_var = self.model.apply({"params": params}, x)
            inv_var = jnp.exp(-log_var)
            mse_loss = jnp.mean(jnp.mean(jnp.square(mu - y) * inv_var, axis=-1), axis=-1)
            var_loss = jnp.mean(jnp.mean(log_var, axis=-1), axis=-1)
            train_loss = mse_loss + var_loss
            return train_loss, {'mse_loss': mse_loss, 'var_loss': var_loss, 'train_loss': train_loss}

        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 1, 1))
        (_, log_info), gradients = grad_fn(model_state.params, batch_inputs, batch_targets)

        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)

        # Update TrainState
        model_state = model_state.apply_gradients(grads=gradients)

        return log_info, model_state


    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   critic_target_params: FrozenDict,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   alpha_state: train_state.TrainState,
                   key: jnp.ndarray):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        observations, actions, rewards, discounts, next_observations = batch
        def loss_fn(actor_params: FrozenDict,
                    critic_params: FrozenDict,
                    alpha_params: FrozenDict,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    discount: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    rng: jnp.ndarray):
            """compute loss for a single transition"""
            rng, rng1, rng2 = jax.random.split(rng, 3)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply({"params": actor_params}, rng1, observation)

            # Alpha loss: stop gradient to avoid affect Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.squeeze(jnp.minimum(sampled_q1, sampled_q2))

            # Actor loss
            alpha = jax.lax.stop_gradient(alpha)  # stop gradient to avoid affect Alpha parameters
            actor_loss = (alpha * logp - sampled_q)

            # Critic loss
            q1, q2 = self.critic.apply({"params": critic_params}, observation, action)
            q1 = jnp.squeeze(q1)
            q2 = jnp.squeeze(q2)

            # Use frozen_actor_params to avoid affect Actor parameters
            _, next_action, logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng2, next_observation)
            next_q1, next_q2 = self.critic.apply({"params": critic_target_params}, next_observation, next_action)

            next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2))
            if self.backup_entropy:
                next_q -= alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2

            # CQL loss
            rng3, rng4 = jax.random.split(rng, 2)
            cql_random_actions = jax.random.uniform(rng3, shape=(self.num_random, self.act_dim))

            # Sample 10 actions with current state
            repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0), repeats=self.num_random, axis=0)
            repeat_next_observations = jnp.repeat(jnp.expand_dims(next_observation, axis=0), repeats=self.num_random, axis=0)
            _, cql_sampled_actions, cql_logp = self.actor.apply({"params": frozen_actor_params}, rng3, repeat_observations)
            _, cql_next_actions, cql_logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng4, repeat_next_observations)

            cql_random_q1, cql_random_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_random_actions)
            cql_q1, cql_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_sampled_actions)
            cql_next_q1, cql_next_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_next_actions)

            random_density = np.log(0.5 ** self.act_dim)
            cql_concat_q1 = jnp.concatenate([jnp.squeeze(cql_random_q1) - random_density,
                                             jnp.squeeze(cql_next_q1) - cql_logp_next_action,
                                             jnp.squeeze(cql_q1) - cql_logp])
            cql_concat_q2 = jnp.concatenate([jnp.squeeze(cql_random_q2) - random_density,
                                             jnp.squeeze(cql_next_q2) - cql_logp_next_action,
                                             jnp.squeeze(cql_q2) - cql_logp])

            cql1_loss = (jax.scipy.special.logsumexp(cql_concat_q1) - q1) * self.min_q_weight
            cql2_loss = (jax.scipy.special.logsumexp(cql_concat_q2) - q2) * self.min_q_weight

            # Loss weight form Dopamine
            total_loss = critic_loss + actor_loss + alpha_loss + cql1_loss + cql2_loss
            log_info = {"critic_loss": critic_loss, "actor_loss": actor_loss, "alpha_loss": alpha_loss,
                        "cql1_loss": cql1_loss, "cql2_loss": cql2_loss, 
                        "q1": q1, "q2": q2, "cql_q1": cql_q1.mean(), "cql_q2": cql_q2.mean(),
                        "cql_next_q1": cql_next_q1.mean(), "cql_next_q2": cql_next_q2.mean(),
                        "random_q1": cql_random_q1.mean(), "random_q2": cql_random_q2.mean(),
                        "alpha": alpha, "logp": logp, "logp_next_action": logp_next_action}
            # if self.with_lagrange:
            #     log_info.update({"cql_alpha": 0.0})

            return total_loss, log_info

        grad_fn = jax.vmap(
            jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
            in_axes=(None, None, None, 0, 0, 0, 0, 0, 0))
        rng = jnp.stack(jax.random.split(key, num=actions.shape[0]))

        (_, log_info), gradients = grad_fn(actor_state.params,
                                           critic_state.params,
                                           alpha_state.params,
                                           observations,
                                           actions,
                                           rewards,
                                           discounts,
                                           next_observations,
                                           rng)

        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)

        actor_grads, critic_grads, alpha_grads = gradients

        # Update TrainState
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        return log_info, actor_state, critic_state, alpha_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def update_target_params(self, params: FrozenDict, target_params: FrozenDict):
        def _update(param, target_param):
            return self.tau * param + (1 - self.tau) * target_param

        updated_params = jax.tree_multimap(_update, params, target_params)
        return updated_params

    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray, eval_mode: bool = False) -> jnp.ndarray:
        observation = jax.device_put(observation[None])
        rng, sample_rng = jax.random.split(rng)
        mean_action, sampled_action, _ = self.actor.apply({"params": params}, sample_rng, observation)
        return rng, jnp.where(eval_mode, mean_action.flatten(), sampled_action.flatten())

    def update_old(self, replay_buffer, batch_size: int = 256):
        self.update_step += 1

        # Sample from the buffer
        batch = replay_buffer.sample(batch_size)
        self.rng, key = jax.random.split(self.rng)
        if self.with_lagrange:
            (log_info, self.actor_state, self.critic_state, self.alpha_state, self.cql_alpha_state) =\
                self.train_step(batch, self.critic_target_params, self.actor_state, self.critic_state,
                                self.alpha_state, self.cql_alpha_state, key)
        else:
            (log_info, self.actor_state, self.critic_state, self.alpha_state) =\
                self.train_step(batch, self.critic_target_params, self.actor_state, self.critic_state,
                                self.alpha_state, key)

        # update target network
        params = self.critic_state.params
        target_params = self.critic_target_params
        self.critic_target_params = self.update_target_params(params, target_params)

        return log_info

    def update(self, replay_buffer, batch_size: int = 256):
        self.update_step += 1
