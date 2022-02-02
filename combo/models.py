from typing import Any, Callable, Optional
from flax import linen as nn
from flax import serialization
from flax.core import FrozenDict
from flax.training import train_state
import functools
import gym
import d4rl
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
from static_fn import static_fns
from tqdm import trange
from utils import Batch, ReplayBuffer, get_training_data


LOG_STD_MAX = 2.
LOG_STD_MIN = -5.

kernel_initializer = jax.nn.initializers.glorot_uniform()


class Actor(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, rng: Any, observation: jnp.ndarray):
        x = nn.relu(nn.Dense(256, kernel_init=kernel_initializer, name="fc1")(observation))
        x = nn.relu(nn.Dense(256, kernel_init=kernel_initializer, name="fc2")(x))
        x = nn.relu(nn.Dense(256, kernel_init=kernel_initializer, name="fc3")(x))
        x = nn.Dense(2 * self.act_dim, kernel_init=kernel_initializer, name="output")(x)

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
    layer_num: int = 3

    @nn.compact
    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observation, action], axis=-1)
        for i in range(self.layer_num):
            x = nn.relu(nn.Dense(self.hid_dim, kernel_init=kernel_initializer, name=f"fc{i+1}")(x))
        q = nn.Dense(1, kernel_init=kernel_initializer, name="output")(x)
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


class GaussianMLP(nn.Module):
    ensemble_num: int
    out_dim: int
    hid_dim: int = 200
    max_log_var: float = 0.5
    min_log_var: float = -10.0

    def setup(self):
        self.l1 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.hid_dim, name="fc1")
        self.l2 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.hid_dim, name="fc2")
        self.l3 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.hid_dim, name="fc3")
        self.l4 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.hid_dim, name="fc4")
        self.l5 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.out_dim*2, name="fc5")

    def __call__(self, x):
        x = nn.swish(self.l1(x))
        x = nn.swish(self.l2(x))
        x = nn.swish(self.l3(x))
        x = nn.swish(self.l4(x))
        x = self.l5(x)

        mu, log_var = jnp.split(x, 2, axis=-1)
        log_var = self.max_log_var - jax.nn.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + jax.nn.softplus(log_var - self.min_log_var)
        return mu, log_var


class DynamicsModel:
    """
    val_loss = 0.455
    val_loss1 = [0.55001634 0.4690651  0.44378504 0.3747698  0.4047458  0.4849194 0.45814472]
    """
    def __init__(self,
                 env: str = "hopper-medium-v2",
                 seed: int = 42,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 holdout_num: int = 1000,
                 lr: float = 1e-3,
                 weight_decay: float = 5e-5,
                 epochs: int = 300,
                 batch_size: int = 1024,
                 max_patience: int = 10,
                 model_dir: str = "./ensemble_models"):

        # Model parameters
        self.seed = seed
        self.lr = lr
        self.static_fn = static_fns[env.split('-')[0].lower()]
        self.weight_decay = weight_decay
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        self.elite_models = None
        self.holdout_num = holdout_num
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_patience = max_patience

        # Environment & ReplayBuffer
        self.env = gym.make(env)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim)
        self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env))

        # Initilaize the ensemble model
        self.save_file = f"{model_dir}/{env}/s{seed}_b{batch_size}"
        self.elite_mask = None

        np.random.seed(seed+10)
        rng = jax.random.PRNGKey(seed+10)
        _, model_key = jax.random.split(rng, 2)
        self.model = GaussianMLP(ensemble_num=ensemble_num, out_dim=obs_dim+1)
        dummy_model_inputs = jnp.ones([ensemble_num, obs_dim+act_dim], dtype=jnp.float32)
        model_params = self.model.init(model_key, dummy_model_inputs)["params"]

        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=model_params,
            tx=optax.adamw(learning_rate=lr, weight_decay=weight_decay))

    def load(self, filename):
        with open(f"{filename}.ckpt", "rb") as f:
            model_params = serialization.from_bytes(
                self.model_state.params, f.read())
        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=model_params,
            tx=optax.adamw(learning_rate=self.lr,
            weight_decay=self.weight_decay))
        elite_idx = np.loadtxt(f'{filename}_elite_models.txt', dtype=np.int32)[:self.elite_num]
        self.elite_mask = jnp.eye(self.ensemble_num)[elite_idx, :]

    def train(self):
        inputs, targets, holdout_inputs, holdout_targets = get_training_data(
            self.replay_buffer, self.ensemble_num, self.holdout_num)

        patience = 0 
        batch_num = int(np.ceil(len(inputs) / self.batch_size))
        min_val_loss = np.inf
        optial_params = None
        res = []

        # Loss functions
        @jax.jit
        def loss_fn(params, x, y):
            """
            x = np.random.normal(size=(7, 128, 14))[:, 0, :]  # (7, 14)
            y = np.random.normal(size=(7, 128, 12))[:, 0, :]  # (7, 12)
            mu, log_var = model.model.apply({"params": model.model_state.params}, x)  # (7, 12), (7, 12)
            """
            mu, log_var = self.model.apply({"params": params}, x)  # (1, ) ==> (7, 256)
            inv_var = jnp.exp(-log_var)  # (7, 12)
            mse_loss = jnp.mean(jnp.mean(jnp.square(mu - y) * inv_var, axis=-1), axis=-1)
            var_loss = jnp.mean(jnp.mean(log_var, axis=-1), axis=-1)
            train_loss = mse_loss + var_loss
            return train_loss, {"mse_loss": mse_loss, "var_loss": var_loss, "train_loss": train_loss}

        @jax.jit
        def val_loss_fn(params, x, y):
            """
            x = np.random.normal(size=(7, 128, 14))[:, 0, :]
            y = np.random.normal(size=(7, 128, 12))[:, 0, :]
            mu, log_var = jax.lax.stop_gradient(model.model.apply({"params": model.model_state.params}, x))  
            """
            mu, log_var = jax.lax.stop_gradient(self.model.apply({"params": params}, x))  # (7, 12), (7, 12)
            inv_var = jnp.exp(-log_var)
            mse_loss = jnp.mean(jnp.square(mu - y), axis=-1)
            reward_loss = jnp.mean(jnp.square(mu[:, 0] - y[:, 0]), axis=-1) 
            state_mse_loss = jnp.mean(jnp.mean(jnp.square(mu[:, 1:] - y[:, 1:]) * inv_var[:, 1:], axis=-1), axis=-1)
            state_var_loss = jnp.mean(jnp.mean(log_var[:, 1:], axis=-1), axis=-1)
            state_loss = state_mse_loss + state_var_loss
            return mse_loss, {"reward_loss": reward_loss, "state_loss": state_loss}

        # Wrap loss functions with jax.vmap
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 1, 1))
        val_loss_fn = jax.vmap(val_loss_fn, in_axes=(None, 1, 1))

        for epoch in trange(self.epochs):
            shuffled_idxs = np.concatenate([np.random.permutation(np.arange(
                inputs.shape[0])).reshape(1, -1) for _ in range(self.ensemble_num)], axis=0)
            train_loss, mse_loss, var_loss = [], [], []
            for i in range(batch_num):
                batch_idxs = shuffled_idxs[:, i*self.batch_size:(i+1)*self.batch_size]
                batch_inputs = inputs[batch_idxs]  # (7, 256, 14)
                batch_targets = targets[batch_idxs]  # (7, 256, 12)

                (_, log_info), gradients = grad_fn(self.model_state.params, batch_inputs, batch_targets)
                log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
                gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
                self.model_state = self.model_state.apply_gradients(grads=gradients)

                train_loss.append(log_info["train_loss"].item())
                mse_loss.append(log_info["mse_loss"].item())
                var_loss.append(log_info["var_loss"].item())

            val_loss, val_info = val_loss_fn(self.model_state.params, holdout_inputs, holdout_targets)
            val_loss = jnp.mean(val_loss, axis=0)  # (7,)
            # val_loss = jnp.mean(val_loss_fn(self.model_state.params, holdout_inputs, holdout_targets), axis=0)  # (7,)
            val_info = jax.tree_map(functools.partial(jnp.mean, axis=0), val_info)
            mean_val_loss = jnp.mean(val_loss)
            if mean_val_loss < min_val_loss:
                optimal_params = self.model_state.params
                min_val_loss = mean_val_loss
                elite_models = jnp.argsort(val_loss)  # find elite models
                patience = 0
            else:
                patience += 1
            if patience == self.max_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

            res.append((epoch, sum(train_loss)/batch_num, sum(mse_loss)/batch_num,
                        sum(var_loss)/batch_num, mean_val_loss))
            print(f"Epoch #{epoch+1}: "
                  f"train_loss = {sum(train_loss)/batch_num:.3f} "
                  f"mse_loss = {sum(mse_loss)/batch_num:.3f} "
                  f"var_loss = {sum(var_loss)/batch_num:.3f} "
                  f"val_loss = {mean_val_loss:.3f} "
                  f"val_rew_loss = {val_info['reward_loss']:.3f} "
                  f"val_state_loss = {val_info['state_loss']:.3f}"
            )

        res_df = pd.DataFrame(res, columns=[
            "epoch", "train_loss", "mse_loss", "var_loss", "val_loss"])
        res_df.to_csv(f"{self.save_file}.csv")
        with open(f"{self.save_file}.ckpt", "wb") as f:
            f.write(serialization.to_bytes(optimal_params))
        with open(f"{self.save_file}_elite_models.txt", "w") as f:
            for idx in elite_models:
                f.write(f"{idx}\n")

    def step(self, key, observations, actions):
        model_idx = jax.random.randint(key, shape=(actions.shape[0],), minval=0, maxval=self.elite_num)
        model_masks = self.elite_mask[model_idx].reshape(-1, self.ensemble_num, 1)

        @jax.jit
        def rollout(params, rng, observation, action, model_mask):
            x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
            model_mu, model_log_var = self.model.apply({"params": params}, x)
            model_std = jnp.sqrt(jnp.exp(model_log_var))  # (7, 12)
            model_noise = model_std * jax.random.normal(rng, (self.ensemble_num, self.obs_dim+1))  # (7, 12)
            reward_noise, observation_noise = jnp.split(model_noise, [1], axis=-1)     # (7, 1), (7, 11)
            reward_mu, observation_mu = jnp.split(model_mu, [1], axis=-1)              # (7, 1), (7, 11)
            model_next_observation = observation + jnp.sum(
                model_mask * (observation_mu + observation_noise), axis=0)              # (1, 11)
            model_reward = jnp.sum(model_mask * (reward_mu + reward_noise), axis=0)   # (1, 1)
            return model_next_observation, model_reward

        rollout = jax.vmap(rollout, in_axes=(None, 0, 0, 0, 0))
        rollout_rng = jnp.stack(jax.random.split(key, num=actions.shape[0]))
        next_observations, rewards = rollout(self.model_state.params, rollout_rng, observations, actions, model_masks)
        terminals = self.static_fn.termination_fn(observations, actions, next_observations)
        return next_observations, rewards.squeeze(), terminals.squeeze()


class COMBOAgent:
    def __init__(self,
                 env: str = "hopper-medium-v2",
                 obs_dim: int = 11,
                 act_dim: int = 3,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 lr_actor: float = 3e-5,
                 auto_entropy_tuning: bool = True,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = False,
                 num_random: int = 10,
                 min_q_weight: float = 3.0,
                 with_lagrange: bool = False,
                 lagrange_thresh: int = 5.0,

                 # COMBO
                 horizon: int = 5,
                 lr_model: float = 1e-3,
                 weight_decay: float = 5e-5,
                 real_ratio: float = 0.5,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 max_patience: int = 5,
                 batch_size: int = 256,
                 rollout_batch_size: int = 10000,
                 holdout_ratio: float = 0.1,
                 model_dir: str = 'ensemble_models'):

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
        self.horizon = horizon
        self.lr_model = lr_model
        self.weight_decay = weight_decay
        self.max_patience = max_patience
        self.holdout_ratio = holdout_ratio
        self.ensemble_num = ensemble_num
        self.real_ratio = real_ratio 

        # Initialize random keys
        self.rng = jax.random.PRNGKey(seed)
        self.rng, self.rollout_rng = jax.random.split(self.rng, 2)
        actor_key, critic_key, model_key = jax.random.split(self.rng, 3)

        # Dummy inputs
        dummy_obs = jnp.ones([1, self.obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, self.act_dim], dtype=jnp.float32)
        dummy_model_inputs = jnp.ones([self.ensemble_num, self.obs_dim+self.act_dim], dtype=jnp.float32)

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
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(self.lr))

        # Initialize the Dynamics Model
        self.model = DynamicsModel(env=env, seed=seed,
                                   ensemble_num=ensemble_num,   
                                   elite_num=elite_num,
                                   model_dir=model_dir)

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
        self.min_q_weight = min_q_weight if not with_lagrange else 1.0
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.rng, cql_key = jax.random.split(self.rng, 2)
            self.log_cql_alpha = Scalar(0.0)  # 1.0
            self.cql_alpha_state = train_state.TrainState.create(
                apply_fn=None,
                params=self.log_cql_alpha.init(cql_key)["params"],
                tx=optax.adam(self.lr_actor))

        # replay buffer
        self.batch_size = batch_size
        self.rollout_batch_size = rollout_batch_size
        self.real_batch_size = int(real_ratio * batch_size)
        self.model_batch_size = batch_size - self.real_batch_size
        self.masks = np.concatenate([np.ones(self.real_batch_size), np.zeros(self.model_batch_size)])

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   observations: jnp.array,
                   actions: jnp.array,
                   rewards: jnp.array,
                   discounts: jnp.array,
                   next_observations: jnp.array,
                   critic_target_params: FrozenDict,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   alpha_state: train_state.TrainState,
                   key: jnp.ndarray):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        def loss_fn(actor_params: FrozenDict,
                    critic_params: FrozenDict,
                    alpha_params: FrozenDict,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    discount: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    mask: jnp.ndarray,
                    rng: jnp.ndarray):
            """compute loss for a single transition"""
            rng, rng1, rng2 = jax.random.split(rng, 3)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply({"params": actor_params}, rng1, observation)

            # Alpha loss: stop gradient to avoid affect Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha)  # stop gradient to avoid affect Alpha parameters

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.squeeze(jnp.minimum(sampled_q1, sampled_q2))

            # Actor loss
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
            critic_loss1 = 0.5 * (q1 - target_q)**2
            critic_loss2 = 0.5 * (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2


            # COMBO CQL loss
            rng3, rng4 = jax.random.split(rng, 2)
            cql_random_actions = jax.random.uniform(
                rng3, shape=(self.num_random, self.act_dim), minval=-1.0, maxval=1.0)

            # repeat next observations
            repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0),
                                             repeats=self.num_random, axis=0)

            # sample actions with actor
            _, cql_sampled_actions, cql_logp = self.actor.apply(
                {"params": frozen_actor_params}, rng3, repeat_observations)

            # random q values
            cql_random_q1, cql_random_q2 = self.critic.apply({"params": critic_params},
                                                             repeat_observations,
                                                             cql_random_actions)

            # cql q values
            cql_q1, cql_q2 = self.critic.apply({"params": critic_params},
                                                repeat_observations,
                                                cql_sampled_actions)

            random_density = np.log(0.5 ** self.act_dim)
            cql_concat_q1 = jnp.concatenate([
                jnp.squeeze(cql_random_q1) - random_density,
                jnp.squeeze(cql_q1) - cql_logp,
            ])
            cql_concat_q2 = jnp.concatenate([
                jnp.squeeze(cql_random_q2) - random_density,
                jnp.squeeze(cql_q2) - cql_logp,
            ])

            ood_q1 = jax.scipy.special.logsumexp(cql_concat_q1)
            ood_q2 = jax.scipy.special.logsumexp(cql_concat_q2)

            # compute logsumexp loss w.r.t model_states 
            cql1_loss = (ood_q1*(1-mask) - q1*mask) * self.min_q_weight / self.real_ratio
            cql2_loss = (ood_q2*(1-mask) - q2*mask) * self.min_q_weight / self.real_ratio

            total_loss = alpha_loss + actor_loss + critic_loss + cql1_loss + cql2_loss
            log_info = {
                "critic_loss1": critic_loss1,
                "critic_loss2": critic_loss2,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "cql1_loss": cql1_loss,
                "cql2_loss": cql2_loss, 
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
                "logp_next_action": logp_next_action
            }

            return total_loss, log_info

        grad_fn = jax.vmap(
            jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
            in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0))
        rng = jnp.stack(jax.random.split(key, num=actions.shape[0]))

        (_, log_info), gradients = grad_fn(actor_state.params,
                                           critic_state.params,
                                           alpha_state.params,
                                           observations,
                                           actions,
                                           rewards,
                                           discounts,
                                           next_observations,
                                           self.masks,
                                           rng)

        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        extra_log_info = {
            'q1_min': log_info['q1'].min(),
            'q1_max': log_info['q1'].max(),
            'q1_std': log_info['q1'].std(),
            'q2_min': log_info['q2'].min(),
            'q2_max': log_info['q2'].max(),
            'q2_std': log_info['q2'].std(),
            'target_q_min': log_info['target_q'].min(),
            'target_q_max': log_info['target_q'].max(),
            'target_q_std': log_info['target_q'].std(),
            'ood_q1_min': log_info['ood_q1'].min(),
            'ood_q1_max': log_info['ood_q1'].max(),
            'ood_q1_std': log_info['ood_q1'].std(),
            'ood_q2_min': log_info['ood_q2'].min(),
            'ood_q2_max': log_info['ood_q2'].max(),
            'ood_q2_std': log_info['ood_q2'].std(),
            'critic_loss_min': log_info['critic_loss'].min(),
            'critic_loss_max': log_info['critic_loss'].max(),
            'critic_loss_std': log_info['critic_loss'].std(),
            'critic_loss1_min': log_info['critic_loss1'].min(),
            'critic_loss1_max': log_info['critic_loss1'].max(),
            'critic_loss1_std': log_info['critic_loss1'].std(),
            'critic_loss2_min': log_info['critic_loss2'].min(),
            'critic_loss2_max': log_info['critic_loss2'].max(),
            'critic_loss2_std': log_info['critic_loss2'].std(),
            'cql1_loss_min': log_info['cql1_loss'].min(),
            'cql1_loss_max': log_info['cql1_loss'].max(),
            'cql1_loss_std': log_info['cql1_loss'].std(),
            'cql2_loss_min': log_info['cql2_loss'].min(),
            'cql2_loss_max': log_info['cql2_loss'].max(),
            'cql2_loss_std': log_info['cql2_loss'].std(),
        }
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        log_info.update(extra_log_info)
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

    def update(self, replay_buffer, model_buffer):
        # rollout the model
        if self.update_step % 1000 == 0:
            observations = replay_buffer.sample(self.rollout_batch_size).observations  # (10000, 11)
            # sample_rng = jnp.stack(jax.random.split(self.rollout_rng, num=self.rollout_batch_size))
            # select_action = jax.vmap(self.select_action, in_axes=(None, 0, 0, None))
            for t in range(self.horizon):
                self.rollout_rng, rollout_key = jax.random.split(self.rollout_rng, 2)

                # random actions
                actions = jax.random.uniform(self.rollout_rng, shape=(len(observations), 3),
                                             minval=-1.0, maxval=1.0)
                # sample actions with policy pi
                # sample_rng, actions = select_action(self.actor_state.params, sample_rng, observations, False)

                next_observations, rewards, dones = self.model.step(rollout_key, observations, actions)
                nonterminal_mask = ~dones
                if nonterminal_mask.sum() == 0:
                    print(f'[ Model Rollout ] Breaking early {nonterminal_mask.shape}')
                    break
                model_buffer.add_batch(observations,
                                       actions,
                                       next_observations,
                                       rewards,
                                       dones)
                observations = next_observations[nonterminal_mask]
                # sample_rng = sample_rng[nonterminal_mask]

        # sample from real & model buffer
        real_batch = replay_buffer.sample(self.real_batch_size)
        model_batch = model_buffer.sample(self.model_batch_size)

        concat_observations = np.concatenate([real_batch.observations, model_batch.observations], axis=0)
        concat_actions = np.concatenate([real_batch.actions, model_batch.actions], axis=0)
        concat_rewards = np.concatenate([real_batch.rewards, model_batch.rewards], axis=0)
        concat_discounts = np.concatenate([real_batch.discounts, model_batch.discounts], axis=0)
        concat_next_observations = np.concatenate([real_batch.next_observations, model_batch.next_observations], axis=0)

        # CQL training with COMBO
        self.rng, key = jax.random.split(self.rng)
        log_info, self.actor_state, self.critic_state, self.alpha_state = self.train_step(
            concat_observations, concat_actions, concat_rewards, concat_discounts,
            concat_next_observations, self.critic_target_params, self.actor_state,
            self.critic_state, self.alpha_state, key
        )

        # upate target network
        params = self.critic_state.params
        target_params = self.critic_target_params
        self.critic_target_params = self.update_target_params(params, target_params)

        self.update_step += 1
        return log_info
