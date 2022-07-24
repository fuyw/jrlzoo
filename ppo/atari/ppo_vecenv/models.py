import functools
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np

import env_utils

class ActorCritic(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name="conv1", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name="conv2", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name="conv3", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=512, name="hidden", dtype=jnp.float32)(x)
        x = nn.relu(x)

        logits = nn.Dense(features=self.act_dim, name="logits", dtype=jnp.float32)(x)
        policy_log_probabilities = nn.log_softmax(logits)
        value = nn.Dense(features=1, name="value", dtype=jnp.float32)(x)

        return policy_log_probabilities, value.squeeze(-1)


class PPOAgent:
    """PPOAgent using vectorized environment."""
    def __init__(self, config, act_dim: int, lr: float):
        self.vf_coeff = config.vf_coeff
        self.entropy_coeff = config.entropy_coeff

        # initialize learner
        self.rng = jax.random.PRNGKey(config.seed)
        dummy_obs = jnp.ones([1, 84, 84, 4])
        self.learner = ActorCritic(act_dim)
        learner_params = self.learner.init(self.rng, dummy_obs)["params"]
        self.learner_state = train_state.TrainState.create(
            apply_fn=ActorCritic.apply,
            params=learner_params,
            tx=optax.adam(lr))

    @functools.partial(jax.jit, static_argnames=("self"))    
    def _sample_action(self, params, observations):
        log_probs, values = self.learner.apply({"params": params}, observations)
        return log_probs, values

    def sample_action(self, observation):
        log_probs, _ = self._sample_action(self.learner_state.params, observation)
        probs = np.exp(np.asarray(log_probs))
        action = np.random.choice(probs.shape[1], p=probs)
        return action

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self, learner_state, batch, clip_param: float):
        def loss_fn(params, observations, actions, old_log_probs, targets, advantages):
            log_probs, values = self.learner.apply({"params": params}, observations)
            probs = jnp.exp(log_probs)
            entropy = jnp.sum(-probs*log_probs, axis=1).mean()
            log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
            ratios = jnp.exp(log_probs_act_taken - old_log_probs)
            pg_loss = ratios * advantages
            clipped_loss = advantages * jax.lax.clamp(1.-clip_param, ratios, 1.+clip_param)

            ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)
            value_loss = jnp.mean(jnp.square(targets - values), axis=0) * self.vf_coeff
            entropy_loss = - entropy * self.entropy_coeff
            total_loss = ppo_loss + value_loss + entropy_loss

            log_info = {"ppo_loss": ppo_loss,
                        "value_loss": value_loss,
                        "entropy_loss": entropy_loss,
                        "total_loss": total_loss}
            return total_loss, log_info
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        observations, actions, old_log_probs, targets, advantages = batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)        
        (_, log_info), grads = grad_fn(
            learner_state.params,
            observations,
            actions,
            old_log_probs,
            targets,
            advantages)
        new_learner_state = learner_state.apply_gradients(grads=grads)
        return new_learner_state, log_info

    def update(self, batch, clip_param):
        self.learner_state, log_info = self.train_step(self.learner_state, batch, clip_param)
        return log_info