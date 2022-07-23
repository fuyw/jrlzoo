import multiprocessing as mp
import functools
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import numpy as np
import ml_collections

import env_utils
from utils import Batch


class ActorCritic(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.
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
        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=512, name="hidden", dtype=jnp.float32)(x)
        x = nn.relu(x)

        logits = nn.Dense(features=self.act_dim,
                          name="logits",
                          dtype=jnp.float32)(x)
        policy_log_probabilities = nn.log_softmax(logits)
        value = nn.Dense(features=1, name="value", dtype=jnp.float32)(x)

        return policy_log_probabilities, value.squeeze(-1)


class RemoteActor:
    """Remote actor in a separate process."""

    def __init__(self, env_name: str, rank: int):
        """Start the remote process and create Pipe() to communicate with it."""
        parent_conn, child_conn = mp.Pipe()
        self.proc = mp.Process(target=self.rcv_action_send_exp,
                               args=(child_conn, env_name, rank))
        self.proc.daemon = True
        self.conn = parent_conn
        self.proc.start()

    def rcv_action_send_exp(self, conn, env_name: str, rank: int = 0):
        """Run remote actor.

        Receive action from the main learner, perform one step of simulation and
        send back collected experience.
        """
        env = env_utils.create_env(env_name,
                                   clip_rewards=True,
                                   seed=rank + 100)
        while True:
            obs, done = env.reset(), False
            while not done:
                # (1) send observation to the learner
                conn.send(obs[None, ...])

                # (2) receive sampled action from the learner
                action = conn.recv()

                # (3) interact with the environment
                next_obs, reward, done, _ = env.step(action)
                experience = (reward, done)

                # (4) send next observation to the learner
                conn.send(experience)
                if done:
                    break
                obs = next_obs


class PPOAgent:
    """PPOAgent adapted from Flax PPO example."""

    def __init__(self, config: ml_collections.ConfigDict, act_dim: int,
                 lr: float):
        self.vf_coeff = config.vf_coeff
        self.entropy_coeff = config.entropy_coeff
        self.clip_param = config.clip_param

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
            RemoteActor(config.env_name, i) for i in range(config.actor_num)
        ]

    @functools.partial(jax.jit, static_argnames=("self"))
    def _sample_actions(self, params: FrozenDict, observations: jnp.ndarray):
        log_probs, values = self.learner.apply({"params": params},
                                               observations)
        return log_probs, values

    def sample_actions(self, observations: jnp.ndarray):
        log_probs, _ = self._sample_actions(self.learner_state.params,
                                            observations)
        probs = np.exp(np.asarray(log_probs))
        actions = np.array(
            [np.random.choice(len(prob), p=prob) for prob in probs])
        return actions

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self, learner_state: train_state.TrainState, batch: Batch):

        def loss_fn(params, observations, actions, old_log_probs, targets,
                    advantages):
            log_probs, values = self.learner.apply({"params": params},
                                                   observations)

            # clipped PPO loss
            log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs,
                                                                actions)
            ratios = jnp.exp(log_probs_act_taken - old_log_probs)
            clipped_ratios = jnp.clip(ratios, 1. - self.clip_param,
                                      1. + self.clip_param)
            actor_loss1 = ratios * advantages
            actor_loss2 = clipped_ratios * advantages
            ppo_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

            # value loss
            value_loss = jnp.square(targets - values).mean() * self.vf_coeff

            # entropy loss
            probs = jnp.exp(log_probs)
            entropy = jnp.sum(-probs * log_probs, axis=1).mean()
            entropy_loss = -entropy * self.entropy_coeff

            # total loss
            total_loss = ppo_loss + value_loss + entropy_loss

            log_info = {
                "ppo_loss": ppo_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "total_loss": total_loss,
                "avg_target": targets.mean(),
                "max_target": targets.max(),
                "min_target": targets.min(),
                "avg_value": values.mean(),
                "max_value": values.max(),
                "min_value": values.min(),
                "avg_ratio": ratios.mean(),
                "max_ratio": ratios.max(),
                "min_ratio": ratios.min(),
                "avg_logp": log_probs.mean(),
                "max_logp": log_probs.max(),
                "min_logp": log_probs.min(),
            }
            return total_loss, log_info

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        normalized_advantages = (batch.advantages - batch.advantages.mean()
                                 ) / (batch.advantages.std() + 1e-8)
        (_, log_info), grads = grad_fn(learner_state.params,
                                       batch.observations, batch.actions,
                                       batch.log_probs, batch.targets,
                                       normalized_advantages)
        new_learner_state = learner_state.apply_gradients(grads=grads)
        return new_learner_state, log_info

    def update(self, batch: Batch):
        self.learner_state, log_info = self.train_step(self.learner_state,
                                                       batch)
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname,
                                    self.learner_state,
                                    cnt,
                                    prefix="ppo_",
                                    keep=20,
                                    overwrite=True)
