from typing import Any, Callable, Optional, Sequence, Tuple
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import functools
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
# from utils import target_update


###################
# Utils Functions #
###################
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


######################
# Actor-Critic Model #
######################
class ActorCritic(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        # actor
        self.actor_net = MLP(self.hidden_dims,
                             init_fn=init_fn(self.initializer),
                             activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim, kernel_init=init_fn(self.initializer))
        self.log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))

        # critic
        self.critic_net = MLP((*self.hidden_dims, 1),
                              init_fn=init_fn(self.initializer),
                              activate_final=False)

    def __call__(self, observation: jnp.ndarray):
        x = self.actor_net(observation)
        mu = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        mean_action = nn.tanh(mu) * self.max_action
        action_distribution = distrax.MultivariateNormalDiag(mean_action, std)
        value = self.critic_net(observation).squeeze(-1)
        return mean_action, action_distribution, value

    def sample(self, observation: jnp.ndarray):
        x = self.actor_net(observation)
        mu = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        mean_action = nn.tanh(mu) * self.max_action
        action_distribution = distrax.MultivariateNormalDiag(mean_action, std)
        return mean_action, action_distribution

    def get_value(self, observation: jnp.ndarray):
        value = self.critic_net(observation).squeeze(-1)
        return value


class A2CAgent:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 max_action: float = 1.0,
                 seed: int = 0,
                 gamma: float = 0.99,
                 lr: float = 3e-4):

        self.gamma = gamma
        self.max_action = max_action

        self.rng = jax.random.PRNGKey(seed)
        self.rng, key = jax.random.split(self.rng, 2)

        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)

        # initialize the actor
        self.model = ActorCritic(act_dim=act_dim)
        params = self.model.init(key, dummy_obs)["params"]
        self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                                   params=params,
                                                   tx=optax.adam(lr))

    @functools.partial(jax.jit, static_argnames=("self"))
    def _get_value(self, params, observation):
        value = self.model.apply({"params": params},
                                 observation,
                                 method=ActorCritic.get_value)
        return value

    def get_value(self, observation):
        value = self._get_value(self.state.params, observation)
        value = np.asarray(value)
        return value

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self, params, rng, observation):
        mean_action, dist = self.model.apply({"params": params},
                                             observation,
                                             method=ActorCritic.sample)
        sampled_action, logp = dist.sample_and_log_prob(seed=rng)
        return mean_action, sampled_action, logp

    def sample_action(self, observation, eval_mode=False):
        self.rng, sample_rng = jax.random.split(self.rng)
        mean_action, sampled_action, _ = self._sample_action(
            self.state.params, sample_rng, observation)
        action = mean_action if eval_mode else sampled_action
        action = np.asarray(action)
        action = action.clip(-self.max_action, self.max_action)
        return action

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self, state, observations, actions, returns):
        def loss_fn(params, observations, actions, returns):
            _, dist, values = self.model.apply({"params": params}, observations)
            log_probs = dist.log_prob(actions)
            advantage = returns - values

            entropy_loss = dist.entropy().mean() 
            actor_loss = -(log_probs * jax.lax.stop_gradient(advantage)).mean()
            critic_loss = jnp.square(advantage).mean()
            total_loss = 0.5 * actor_loss + 0.5 * critic_loss - 0.001 * entropy_loss

            return total_loss, {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy_loss": entropy_loss
            }
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, argnums=(0), has_aux=True),
                           in_axes=(None, 0, 0, 0))
        (_, log_info), grads = grad_fn(state.params, observations, actions, returns)
        grads = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), grads)
        log_info = jax.tree_util.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        new_state = state.apply_gradients(grads=grads)
        return new_state, log_info

    def update(self, observations, actions, returns):
        self.state, log_info = self.train_step(self.state, observations, actions, returns)
        return log_info

    def compute_returns(self, next_value, rewards, masks):
        R = next_value
        L = len(rewards)
        returns = np.zeros((L, len(rewards[0])))
        for step in range(L-1, -1, -1):
            R = rewards[step] + self.gamma * R * masks[step]
            returns[step] = R
        return returns
