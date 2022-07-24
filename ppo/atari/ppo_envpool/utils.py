from typing import List
import collections
import logging
import functools
import optax
import jax
import jax.numpy as jnp
import numpy as np

ExpTuple = collections.namedtuple(
    "ExpTuple",
    ["observation", "action", "reward", "value", "log_prob", "done"])
AsyncExpTuple = collections.namedtuple(
    "ExpTuple",
    ["observation", "action", "reward", "value", "log_prob", "done", "env_id"])
Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "log_probs", "targets", "advantages"])

def get_lr_scheduler(config, loop_steps, iterations_per_step):
    # set lr scheduler
    if config.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(init_value=config.lr,
                                   end_value=0.,
                                   transition_steps=loop_steps *
                                   config.num_epochs * iterations_per_step)
    else:
        lr = config.lr
    return lr


def get_logger(fname):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(rewards: np.ndarray, discounts: np.ndarray,
                   values: np.ndarray, gamma: float, lmbda: float):
    """Use GAE to compute advantages.

    As defined by Eq. (11-12) in PPO paper. Implementation uses key observations:
        A_{t} = delta_t + gamma * lmbda * A_{t+1},
              = delta_t + (gamma * lmbda) * delta_{t+1} + + (gamma * lmbda)^2 * delta_{t+2} + ...
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    """
    assert rewards.shape[0] + 1 == values.shape[0], ("One more value needed.")
    advantages = []
    gae = 0.
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal observations.
        value_diff = gamma * values[t + 1] * discounts[t] - values[t]
        delta = rewards[t] + value_diff

        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + gamma * lmbda * discounts[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)


class PPOBuffer:

    def __init__(self,
                 rollout_len,
                 actor_num,
                 gamma,
                 lmbda,
                 obs_shape=(84, 84, 4)):
        self.observations = np.zeros((rollout_len, actor_num, *obs_shape),
                                     dtype=np.float32)
        self.actions = np.zeros((rollout_len, actor_num), dtype=np.int32)
        self.rewards = np.zeros((rollout_len, actor_num), dtype=np.float32)
        self.values = np.zeros((rollout_len + 1, actor_num), dtype=np.float32)
        self.log_probs = np.zeros((rollout_len, actor_num), dtype=np.float32)
        self.discounts = np.zeros((rollout_len, actor_num), dtype=np.float32)

        self.ptr = 0
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.lmbda = lmbda
        self.actor_num = actor_num
        self.trajectory_len = actor_num * rollout_len
        self.obs_shape = obs_shape

    def add(self, experience: List[ExpTuple]):
        for actor_id, actor_exp in enumerate(experience):
            self.observations[self.ptr, actor_id, ...] = actor_exp.observation
            self.actions[self.ptr, actor_id] = actor_exp.action
            self.rewards[self.ptr, actor_id] = actor_exp.reward
            self.values[self.ptr, actor_id] = actor_exp.value
            self.log_probs[self.ptr, actor_id] = actor_exp.log_prob
            self.discounts[self.ptr, actor_id] = float(not actor_exp.done)
        self.ptr += 1

    def add_experiences(self, experiences: List[List[ExpTuple]]):
        assert len(experiences) == self.rollout_len + 1
        for experience in experiences[:-1]:
            self.add(experience)
        # experience[-1] for next_values
        for a in range(self.actor_num):
            self.values[-1, a] = experiences[-1][a].value
        self.ptr = 0

    def process_experience(self):
        # compute GAE advantage
        advantages = gae_advantages(self.rewards, self.discounts, self.values,
                                    self.gamma, self.lmbda)
        targets = advantages + self.values[:-1, :]

        # concatenate results
        trajectory_batch = Batch(observations=self.observations.reshape(
            (self.trajectory_len, *self.obs_shape)),
                                 actions=self.actions.reshape(-1),
                                 log_probs=self.log_probs.reshape(-1),
                                 targets=targets.reshape(-1),
                                 advantages=advantages.reshape(-1))
        return trajectory_batch


class AsyncPPOBuffer:

    def __init__(self,
                 rollout_len,
                 actor_num,
                 gamma,
                 lmbda,
                 obs_shape=(84, 84, 4)):
        self.observations = np.zeros((rollout_len+1, actor_num, *obs_shape),
                                     dtype=np.float32)
        self.actions = np.zeros((rollout_len+1, actor_num), dtype=np.int32)
        self.rewards = np.zeros((rollout_len+1, actor_num), dtype=np.float32)
        self.values = np.zeros((rollout_len+1, actor_num), dtype=np.float32)
        self.log_probs = np.zeros((rollout_len+1, actor_num), dtype=np.float32)
        self.discounts = np.zeros((rollout_len+1, actor_num), dtype=np.float32)

        self.ptr = 0
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.lmbda = lmbda
        self.actor_num = actor_num
        self.trajectory_len = actor_num * rollout_len
        self.obs_shape = obs_shape

    def add_async(self, experience: List[AsyncExpTuple]):
        for actor_exp in experience:
            actor_id = actor_exp.env_id
            self.observations[self.ptr, actor_id, ...] = actor_exp.observation
            self.actions[self.ptr, actor_id] = actor_exp.action
            self.rewards[self.ptr, actor_id] = actor_exp.reward
            self.values[self.ptr, actor_id] = actor_exp.value
            self.log_probs[self.ptr, actor_id] = actor_exp.log_prob
            self.discounts[self.ptr, actor_id] = float(not actor_exp.done)
        self.ptr += 1

    def add_experiences(self, experiences: List[List[ExpTuple]]):
        assert len(experiences) == (self.rollout_len + 1)*2, f"Get {len(experiences)} experiences"
        for experience in experiences:
            self.add_async(experience)
        self.ptr = 0

    def process_experience(self):
        # compute GAE advantage
        advantages = gae_advantages(self.rewards[1:], self.discounts[1:], self.values,
                                    self.gamma, self.lmbda)
        targets = advantages + self.values[:-1, :]

        # concatenate results
        trajectory_batch = Batch(
            observations=self.observations[:-1].reshape((self.trajectory_len, *self.obs_shape)),
            actions=self.actions[:-1].reshape(-1),
            log_probs=self.log_probs[:-1].reshape(-1),
            targets=targets.reshape(-1),
            advantages=advantages.reshape(-1)
        )
        return trajectory_batch
