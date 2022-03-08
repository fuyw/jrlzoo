from typing import Any, Callable, Tuple, Optional
import functools

from flax import linen as nn
from flax import serialization
from flax.core import frozen_dict
from flax.training import train_state
import gym
import d4rl
import jax
import jax.numpy as jnp
import numpy as np
import optax
from static_fn import static_fns
from utils import Batch, ReplayBuffer, get_training_data

kernel_initializer = jax.nn.initializers.glorot_uniform()


class Actor(nn.Module):
    act_dim: int
    max_action: float

    def setup(self):
        self.l1 = nn.Dense(256, kernel_init=kernel_initializer, name="fc1")
        self.l2 = nn.Dense(256, kernel_init=kernel_initializer, name="fc2")
        self.l3 = nn.Dense(self.act_dim, kernel_init=kernel_initializer, name="fc3")

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(self.l1(observations))
        x = nn.relu(self.l2(x))
        actions = self.max_action * nn.tanh(self.l3(x))
        return actions


class Critic(nn.Module):
    def setup(self):
        self.l1 = nn.Dense(256, kernel_init=kernel_initializer, name="fc1")
        self.l2 = nn.Dense(256, kernel_init=kernel_initializer, name="fc2")
        self.l3 = nn.Dense(1, kernel_init=kernel_initializer, name="fc3")

        self.l4 = nn.Dense(256, kernel_init=kernel_initializer, name="fc4")
        self.l5 = nn.Dense(256, kernel_init=kernel_initializer, name="fc5")
        self.l6 = nn.Dense(1, kernel_init=kernel_initializer, name="fc6")

    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observations, actions], axis=-1)

        q1 = nn.relu(self.l1(x))
        q1 = nn.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = nn.relu(self.l4(x))
        q2 = nn.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, observations: jnp.ndarray,
           actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        q1 = nn.relu(self.l1(x))
        q1 = nn.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def Repr(self, observations: jnp.ndarray,
             actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        q1 = nn.relu(self.l1(x))
        repr = nn.relu(self.l2(q1))
        q1 = self.l3(repr)
        return repr, q1


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
        self.mean_and_logvar = EnsembleDense(ensemble_num=self.ensemble_num, features=self.out_dim*2, name="output")

    def __call__(self, x):
        x = nn.leaky_relu(self.l1(x))
        x = nn.leaky_relu(self.l2(x))
        x = nn.leaky_relu(self.l3(x))
        x = self.mean_and_logvar(x)

        mu, log_var = jnp.split(x, 2, axis=-1)
        log_var = self.max_log_var - jax.nn.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + jax.nn.softplus(log_var - self.min_log_var)
        return mu, log_var


class DynamicsModel:
    def __init__(self,
                 env_name: str = "hopper-medium-v2",
                 seed: int = 42,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 holdout_num: int = 1000,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 epochs: int = 200,
                 batch_size: int = 2048,
                 max_patience: int = 10,
                 model_dir: str = "./saved_dynamics_models"):

        # Model parameters
        self.seed = seed
        self.lr = lr
        self.static_fn = static_fns[env_name.split('-')[0].lower()]
        print(f'Load static_fn: {self.static_fn}')
        self.weight_decay = weight_decay
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        self.elite_models = None
        self.holdout_num = holdout_num
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_patience = max_patience

        # Environment & ReplayBuffer
        print(f'Loading data for {env_name}')
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim)
        self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env))

        # Initilaize saving settings
        self.save_file = f"{model_dir}/{env_name}"
        self.elite_mask = np.eye(self.ensemble_num)[range(elite_num), :]

        # Initilaize the ensemble model
        np.random.seed(seed+10)
        rng = jax.random.PRNGKey(seed+10)
        _, model_key = jax.random.split(rng, 2)
        self.model = GaussianMLP(ensemble_num=ensemble_num, out_dim=self.obs_dim+1)
        dummy_model_inputs = jnp.ones([ensemble_num, self.obs_dim+self.act_dim], dtype=jnp.float32)
        model_params = self.model.init(model_key, dummy_model_inputs)["params"]

        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=model_params,
            tx=optax.adamw(learning_rate=self.lr, weight_decay=self.weight_decay))

        # Normalize inputs
        self.obs_mean = None
        self.obs_std = None 

    def load(self, filename):
        with open(f"{filename}/dynamics_model.ckpt", "rb") as f:
            model_params = serialization.from_bytes(
                self.model_state.params, f.read())
        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=model_params,
            tx=optax.adamw(learning_rate=self.lr,
            weight_decay=self.weight_decay))
        # elite_idx = np.loadtxt(f'{filename}/elite_models.txt', dtype=np.int32)[:self.elite_num]
        # self.elite_mask = np.eye(self.ensemble_num)[elite_idx, :]
        normalize_stat = np.load(f'{filename}/normalize_stat.npz')
        self.obs_mean = normalize_stat['obs_mean']
        self.obs_std = normalize_stat['obs_std']

    def train(self):
        (inputs, targets, holdout_inputs, holdout_targets, self.obs_mean,
         self.obs_std) = get_training_data(self.replay_buffer, self.ensemble_num)
        patience = 0 
        batch_num = int(np.ceil(len(inputs) / self.batch_size))
        min_val_loss = np.inf
        optial_params = None
        res = []
        print(f'batch_num     = {batch_num}')
        print(f'inputs.shape  = {inputs.shape}')
        print(f'targets.shape = {targets.shape}') 
        print(f'holdout_inputs.shape  = {holdout_inputs.shape}')
        print(f'holdout_targets.shape = {holdout_targets.shape}') 

        # Loss functions
        @jax.jit
        def loss_fn(params, x, y):
            mu, log_var = self.model.apply({"params": params}, x)  # (7, 14) ==> (7, 12)
            inv_var = jnp.exp(-log_var)    # (7, 12)
            mse_loss = jnp.square(mu - y)  # (7, 12)
            train_loss = jnp.mean(mse_loss * inv_var + log_var, axis=-1).sum()
            return train_loss, {"mse_loss": mse_loss.mean(),
                                "var_loss": log_var.mean(),
                                "train_loss": train_loss}

        @jax.jit
        def val_loss_fn(params, x, y):
            mu, _ = jax.lax.stop_gradient(self.model.apply({"params": params}, x))  # (7, 14) ==> (7, 12)
            mse_loss = jnp.mean(jnp.square(mu - y), axis=-1)  # (7, 12) ==> (7,)
            reward_loss = jnp.square(mu[:, -1] - y[:, -1]).mean()
            state_loss = jnp.square(mu[:, :-1] - y[:, :-1]).mean()
            return mse_loss, {"reward_loss": reward_loss, "state_loss": state_loss}

        # Wrap loss functions with jax.vmap
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 1, 1))
        val_loss_fn = jax.vmap(val_loss_fn, in_axes=(None, 1, 1))

        for epoch in trange(self.epochs):
            shuffled_idxs = np.concatenate([np.random.permutation(np.arange(
                inputs.shape[0])).reshape(1, -1) for _ in range(self.ensemble_num)], axis=0)  # (7, 1000000)
            train_loss, mse_loss, var_loss = [], [], []
            for i in range(batch_num):
                batch_idxs = shuffled_idxs[:, i*self.batch_size:(i+1)*self.batch_size]
                batch_inputs = inputs[batch_idxs]    # (7, 256, 14)
                batch_targets = targets[batch_idxs]  # (7, 256, 12)

                (_, log_info), gradients = grad_fn(self.model_state.params, batch_inputs, batch_targets)
                log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
                gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
                self.model_state = self.model_state.apply_gradients(grads=gradients)

                train_loss.append(log_info["train_loss"].item())
                mse_loss.append(log_info["mse_loss"].item())
                var_loss.append(log_info["var_loss"].item())

            val_loss, val_info = val_loss_fn(self.model_state.params, holdout_inputs, holdout_targets)
            val_loss = jnp.mean(val_loss, axis=0)  # (N, 7) ==> (7,)
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
                  f"train_loss={sum(train_loss)/batch_num:.3f}\t"
                  f"mse_loss={sum(mse_loss)/batch_num:.3f}\t"
                  f"var_loss={sum(var_loss)/batch_num:.3f}\t"
                  f"val_loss={mean_val_loss:.3f}\t"
                  f"val_rew_loss={val_info['reward_loss']:.3f}\t"
                  f"val_state_loss={val_info['state_loss']:.3f}")

        res_df = pd.DataFrame(res, columns=[
            "epoch", "train_loss", "mse_loss", "var_loss", "val_loss"])
        res_df.to_csv(f"{self.save_file}.csv")
        with open(f"{self.save_file}/dynamics_model.ckpt", "wb") as f:
            f.write(serialization.to_bytes(optimal_params))
        with open(f"{self.save_file}/elite_models.txt", "w") as f:
            for idx in elite_models:
                f.write(f"{idx}\n")

        elite_idx = elite_models.to_py()[:self.elite_num]
        self.elite_mask = np.eye(self.ensemble_num)[elite_idx, :]
        print(f'Elite mask is:\n{self.elite_mask}')
        np.savez(f"{self.save_file}/normalize_stat",
                 obs_mean=self.obs_mean, obs_std=self.obs_std)

    def step(self, key, observations, actions):
        model_idx = jax.random.randint(key, shape=(actions.shape[0],), minval=0, maxval=self.elite_num)
        model_masks = self.elite_mask[model_idx].reshape(-1, self.ensemble_num, 1)

        @jax.jit
        def rollout(params, rng, observation, action, model_mask):
            x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
            model_mu, model_log_var = self.model.apply({"params": params}, x)
            observation_mu, reward_mu = jnp.split(model_mu, [self.obs_dim], axis=-1)

            # model_std = jnp.sqrt(jnp.exp(model_log_var))  # (7, 12)
            # model_noise = model_std * jax.random.normal(rng, (self.ensemble_num, self.obs_dim+1))  # (7, 12)
            # observation_noise, reward_noise = jnp.split(model_noise, [self.obs_dim], axis=-1)

            # model_next_observation = observation + jnp.sum(
            #     model_mask * (observation_mu + observation_noise), axis=0)
            # model_reward = jnp.sum(model_mask * (reward_mu + reward_noise), axis=0)

            # model_next_observation = observation + jnp.sum(model_mask * observation_mu, axis=0)
            # model_reward = jnp.sum(model_mask * reward_mu, axis=0)

            model_next_observation = observation + jnp.sum(model_mask * observation_mu, axis=0)
            model_reward = jnp.sum(model_mask * reward_mu, axis=0)
            return model_next_observation, model_reward

        rollout = jax.vmap(rollout, in_axes=(None, 0, 0, 0, 0))
        rollout_rng = jnp.stack(jax.random.split(key, num=actions.shape[0]))
        next_observations, rewards = rollout(self.model_state.params, rollout_rng, observations, actions, model_masks)
        next_observations = self.denormalize(next_observations)
        terminals = self.static_fn.termination_fn(observations, actions, next_observations)
        return next_observations, rewards.squeeze(), terminals.squeeze()

    def step2(self, key, observations, actions):
        @jax.jit
        def rollout(params, observation, action):
            x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
            model_mu, _ = self.model.apply({"params": params}, x)
            observation_mu, reward_mu = jnp.split(model_mu, [self.obs_dim], axis=-1)
            model_next_observation = observation + observation_mu 
            return model_next_observation, reward_mu
        rollout = jax.vmap(rollout, in_axes=(None, 0, 0))
        next_observations, rewards = rollout(self.model_state.params, observations, actions)
        return next_observations, rewards.squeeze()

    def normalize(self, observations):
        assert (self.obs_mean is not None) and (self.obs_std is not None)
        new_observations = (observations - self.obs_mean) / self.obs_std
        return new_observations

    def denormalize(self, observations):
        assert (self.obs_mean is not None) and (self.obs_std is not None)
        new_observations = observations * self.obs_std + self.obs_mean
        return new_observations


class TD3BCM_Agent:
    def __init__(self,
                 env_name: str,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 noise_clip: float = 0.5,
                 policy_noise: float = 0.2,
                 policy_freq: int = 2,
                 learning_rate: float = 3e-4,
                 alpha: float = 2.5,
                 seed: int = 42,
                 horizon: int = 3,
                 mu: Any = None,
                 std: Any = None):

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.mu = mu
        self.std = std
        self.horizon = horizon

        rng = jax.random.PRNGKey(seed)
        self.actor_rng, self.critic_rng, self.rollout_rng = jax.random.split(rng, 3)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Initialize the actor
        self.actor = Actor(act_dim, max_action)
        actor_params = self.actor.init(self.actor_rng, dummy_obs)["params"]
        self.actor_target_params = actor_params
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=learning_rate))

        # Initialize the critic
        self.critic = Critic()
        critic_params = self.critic.init(self.critic_rng, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=Critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=learning_rate))

        self.update_step = 0

        # Model
        self.model = DynamicsModel(env_name=env_name,
                                   seed=seed,
                                   ensemble_num=7,   
                                   elite_num=5,
                                   model_dir='saved_dynamcis_models')

    # def normalize(inputs, normalization_type, name=None):
    #   if normalization_type == "no_norm":
    #     return inputs
    #   elif normalization_type == 'layer_norm':
    #     return nn.LayerNorm(inputs, bias=True, scale=False, name=name)
    #   elif normalization_type == 'batch_norm':
    #     return inputs

    @functools.partial(jax.jit, static_argnames=("self"))
    def actor_train_step(self, real_batch: Batch, model_batch: Batch,
                         actor_state: train_state.TrainState,
                         critic_state: train_state.TrainState):
        def loss_fn(actor_params, critic_params):
            actions = self.actor.apply({"params": actor_params},
                                       real_batch.observations)      # (B, act_dim)
            q_val = self.critic.apply({"params": critic_params},
                                      real_batch.observations,
                                      actions, method=Critic.Q1)     # (B, 1)
            lmbda = self.alpha / jnp.abs(jax.lax.stop_gradient(q_val)).mean()       # ()
            bc_loss = jnp.mean((actions - real_batch.actions)**2)                   # ()
            actor_loss = -lmbda * jnp.mean(q_val) + bc_loss                         # ()
            return actor_loss, {"actor_loss": actor_loss, "bc_loss": bc_loss}

        (actor_loss, actor_info), actor_grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True)(actor_state.params, critic_state.params)

        # Update Actor TrainState
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        return actor_info, actor_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def critic_train_step(self, batch: Batch, critic_key: Any,
                          critic_state: train_state.TrainState,
                          actor_target_params: frozen_dict.FrozenDict,
                          critic_target_params: frozen_dict.FrozenDict):

        # Add noise to actions
        noise = jax.random.normal(critic_key, batch.actions.shape) * self.policy_noise             # (B, act_dim)
        noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)
        next_actions = self.actor.apply({"params": actor_target_params}, batch.next_observations)  # (B, act_dim)
        next_actions = jnp.clip(next_actions + noise, -self.max_action, self.max_action)

        # Compute the target Q value
        next_q1, next_q2 = self.critic.apply({"params": critic_target_params},
                                             batch.next_observations, next_actions)                # (B, 1), (B, 1)
        next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2))                                        # (B,)
        target_q = batch.rewards + self.gamma * batch.discounts * next_q                           # (B,)

        def loss_fn(critic_params: frozen_dict.FrozenDict, batch: Batch):
            q1, q2 = self.critic.apply({"params": critic_params},
                                       batch.observations, batch.actions)
            q1 = jnp.squeeze(q1)
            q2 = jnp.squeeze(q2)
            critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
                "target_q": target_q.mean()
            }

        (critic_loss, critic_info), critic_grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True)(critic_state.params, batch)

        #  update Critic TrainState
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def update_target_params(self, params: frozen_dict.FrozenDict,
                             target_params: frozen_dict.FrozenDict):
        def _update(param, target_param):
            return self.tau * param + (1 - self.tau) * target_param

        updated_params = jax.tree_multimap(_update, params, target_params)
        return updated_params

    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action(self, params: frozen_dict.FrozenDict,
                      observations: np.ndarray) -> jnp.ndarray:
        actions = self.actor.apply({"params": params}, observations)
        return actions

    def train(self, replay_buffer, model_buffer):
        if self.update_step % 1000 == 0:
            observations = replay_buffer.sample(10000).observations * self.std + self.mu
            for _ in range(self.horizon):
                self.rollout_rng, rollout_key = jax.random.split(self.rollout_rng, 2)
                actions = self.select_action(self.actor_state.params, observations)
                normalized_observations = self.model.normalize(observations)
                next_observations, rewards, dones = self.model.step(rollout_key,
                    normalized_observations, actions)
                nonterminal_mask = ~dones
                model_buffer.add_batch((observations-self.mu)/self.std,
                                       actions,
                                       (next_observations-self.mu)/self.std,
                                       rewards,
                                       dones)
                if nonterminal_mask.sum() == 0:
                    print(f'[ Model Rollout ] Breaking early {nonterminal_mask.shape}')
                    break
                observations = next_observations[nonterminal_mask]

        self.update_step += 1

        # sample from real & model buffer
        real_batch = replay_buffer.sample(128)
        model_batch = model_buffer.sample(128)
        concat_observations = np.concatenate([real_batch.observations, model_batch.observations], axis=0)
        concat_actions = np.concatenate([real_batch.actions, model_batch.actions], axis=0)
        concat_rewards = np.concatenate([real_batch.rewards, model_batch.rewards], axis=0)
        concat_discounts = np.concatenate([real_batch.discounts, model_batch.discounts], axis=0)
        concat_next_observations = np.concatenate([real_batch.next_observations, model_batch.next_observations], axis=0)
        batch = Batch(observations=concat_observations,
                      actions=concat_actions,
                      rewards=concat_rewards,
                      discounts=concat_discounts,
                      next_observations=concat_next_observations)

        # Critic update
        self.critic_rng, critic_key = jax.random.split(self.critic_rng)
        log_info, self.critic_state = self.critic_train_step(
            batch, critic_key, self.critic_state, self.actor_target_params,
            self.critic_target_params)

        # Delayed policy update
        if self.update_step % self.policy_freq == 0:
            actor_info, self.actor_state = self.actor_train_step(
                real_batch, batch, self.actor_state, self.critic_state)
            log_info.update(actor_info)

            # update target network
            params = (self.actor_state.params, self.critic_state.params)
            target_params = (self.actor_target_params,
                             self.critic_target_params)
            updated_params = self.update_target_params(params, target_params)
            self.actor_target_params, self.critic_target_params = updated_params

        log_info['real_batch_rewards'] = real_batch.rewards.sum()
        log_info['real_batch_actions'] = abs(real_batch.actions).reshape(-1).sum()
        log_info['real_batch_discounts'] = real_batch.discounts.sum()
        log_info['model_batch_rewards'] = model_batch.rewards.sum()
        log_info['model_batch_actions'] = abs(model_batch.actions).reshape(-1).sum()
        log_info['model_batch_discounts'] = model_batch.discounts.sum()
        log_info['model_buffer_size'] = model_buffer.size
        log_info['model_buffer_ptr'] = model_buffer.ptr

        return log_info 

    def save(self, filename):
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'wb') as f:
            f.write(serialization.to_bytes(self.critic_state.params))
        actor_file = filename + '_actor.ckpt'
        with open(actor_file, 'wb') as f:
            f.write(serialization.to_bytes(self.actor_state.params))

    def load(self, filename):
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'rb') as f:
            critic_params = serialization.from_bytes(
                self.critic_state.params, f.read())
        self.critic_state = train_state.TrainState.create(
            apply_fn=Critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=self.learning_rate))

        actor_file = filename + '_actor.ckpt'
        with open(actor_file, 'rb') as f:
            actor_params = serialization.from_bytes(
                self.actor_state.params, f.read())
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=self.learning_rate)
        )
