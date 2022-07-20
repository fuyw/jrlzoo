import multiprocessing as mp
import distrax
import functools
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np

import env_utils


#################
# Remote Worker #
#################
class RemoteActor:
    """Remote actor in a separate process."""

    def __init__(self, env_name: str, rank: int):
        """Start remote process and communicate with Pipe()"""
        parent_conn, child_conn = mp.Pipe()
        self.process = mp.Process(target=self.rcv_action_send_exp,
                                  args=(child_conn, env_name, rank))
        self.process.daemon = True
        self.conn = parent_conn
        self.process.start()

    def rcv_action_send_exp(self, conn, env_name: str, rank: int = 0):
        """Run remote actor.

        Receive action from the main learner, perform one step
        simulation and send back collected experience.
        """
        env = env_utils.create_env(env_name,
                                   clip_rewards=True,
                                   seed=rank + 100)
        while True:
            obs = env.reset()
            done = False
            obs = obs[None, ...]  # add batch dimension
            while not done:
                # (1) send observation to the learner
                conn.send(obs)

                # (2) receive sampled action from the learner
                action = conn.recv()

                # (3) interact with the environment
                obs, reward, done, _ = env.step(action)
                next_obs = obs[None, ...] if not done else None
                experience = (obs, action, reward, done)

                # (4) send next observation to the learner
                conn.send(experience)
                if done:
                    break
                obs = next_obs


###################
# Central Learner #
###################
class ActorCritic(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, x):
        """Define the convolutional network architecture.

        Network is used to both estimate policy (logits) and expected
        state value; in other words, hidden layers' params are shared
        between policy and value networks.
        """

        x = x.astype(jnp.float32) / 255.  # (256, 84, 84, 4)
        x = nn.Conv(features=32,
                    kernel_size=(8, 8),
                    strides=(4, 4),
                    name="conv1",
                    dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    name="conv2",
                    dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    name="conv3",
                    dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512, name="hidden", dtype=jnp.float32)(x)

        # hidden representation
        x = nn.relu(x)

        # value function
        value = nn.Dense(features=1, name="value", dtype=jnp.float32)(x)

        # pi distribution
        logits = nn.Dense(features=self.act_dim,
                          name="logits",
                          dtype=jnp.float32)(x)
        action_distribution = distrax.Categorical(logits=logits)
        return action_distribution, value.squeeze(-1)


#############
# PPO Agent #
#############
class PPOAgent:
    """PPOAgent adapted from Flax PPO example."""
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
        self.actors = [
            RemoteActor(config.env_name, i) for i in range(config.num_agents)
        ]

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_action(self, params, key, observations):
        action_distributions, values = self.learner.apply({"params": params},
                                                          observations)
        sampled_actions, log_probs = action_distributions.sample_and_log_prob(
            seed=key)
        return sampled_actions, values, log_probs

    def sample_action(self, observations):
        self.rng, key = jax.random.split(self.rng, 2)
        sampled_actions, values, log_probs = self._sample_action(
            self.learner_state.params, key, observations)
        # TODO: jax.device_get([sampled_actions, values])
        sampled_actions = np.asarray(sampled_actions)
        values = np.asarray(values)
        log_probs = np.asarray(log_probs)
        return sampled_actions, values, log_probs

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self, learner_state, batch, clip_param: float):
        def loss_fn(params, observations, actions, old_log_probs, targets,
                    advantages):
            action_distributions, values = self.learner.apply(
                {"params": params}, observations)

            # value loss
            value_loss = jnp.square(targets - values).mean()

            # entropy loss
            entropy = action_distributions.entropy().mean()

            # policy loss
            log_probs = action_distributions.log_prob(actions)
            ratios = jnp.exp(log_probs - old_log_probs)
            pg_loss = ratios * advantages
            clipped_loss = jnp.clip(ratios, 1. - clip_param,
                                    1. + clip_param) * advantages
            ppo_loss = -jnp.minimum(pg_loss, clipped_loss).mean()

            # total loss
            total_loss = ppo_loss + self.vf_coeff * value_loss -\
                self.entropy_coeff * entropy
            log_info = {
                "value_loss": value_loss,
                "pg_loss": pg_loss.mean(),
                "clipped_loss": clipped_loss.mean(),
                "total_loss": total_loss,
                "entropy_loss": entropy,
                "log_prob_ratio": ratios.mean(),
            }
            return total_loss, log_info

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        observations, actions, old_log_probs, targets, advantages = batch
        # GAE advantage normalization (following the OpenAI baselines).
        advantages = (advantages - advantages.mean()) / (advantages.std() +
                                                         1e-8)
        (_, log_info), grads = grad_fn(
            learner_state.params,
            observations,  # (256, 84, 84, 4)
            actions,  # (256,)
            old_log_probs,  # (256,)
            targets,  # (256,)
            advantages)

        # update TrainState
        new_learner_state = learner_state.apply_gradients(grads=grads)
        return new_learner_state, log_info

    def update(self, batch, clip_param):
        self.learner_state, log_info = self.train_step(self.learner_state,
                                                       batch, clip_param)
        return log_info
