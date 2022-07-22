from typing import List, Tuple
import collections
import logging
import functools
import optax
import jax
import jax.numpy as jnp
import numpy as np


ExpTuple = collections.namedtuple("ExpTuple", ["observation", "action", "reward", "value", "log_prob", "done"])


def get_lr_scheduler(config, loop_steps, iterations_per_step):
    # set lr scheduler
    if config.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(init_value=config.lr,
                                   end_value=0.,
                                   transition_steps=loop_steps*config.num_epochs*iterations_per_step)
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
def gae_advantages(rewards: np.ndarray,
                   terminal_masks: np.ndarray,
                   values: np.ndarray,
                   discount: float,
                   gae_param: float):
    """Use GAE to compute advantages.

    As defined by Eq. (11-12) in PPO paper. Implementation uses key observation
    that A_{t} = delta_t + gamma * lmbda * A_{t+1}.
    """
    assert rewards.shape[0] + 1 == values.shape[0], ("One more value needed.")
    advantages = []
    gae = 0.
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal observations.
        value_diff = discount * values[t+1] * terminal_masks[t] - values[t]
        delta = rewards[t] + value_diff

        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + discount * gae_param * terminal_masks[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)


class PPOBuffer:
    def __init__(self, size, num_agents, gamma, lmbda, obs_shape=(84, 84, 4)):
        self.observations = np.zeros((size, num_agents, *obs_shape), dtype=np.float32)
        self.actions   = np.zeros((size, num_agents), dtype=np.int32)
        self.rewards   = np.zeros((size, num_agents), dtype=np.float32)
        self.values    = np.zeros((size+1, num_agents), dtype=np.float32)
        self.log_probs = np.zeros((size, num_agents), dtype=np.float32)
        self.dones     = np.zeros((size, num_agents), dtype=np.float32)

        self.ptr = 0
        self.size = size
        self.gamma = gamma
        self.lmbda = lmbda
        self.num_agents = num_agents
        self.trajectory_len = num_agents * size

    def add(self, experience: List[ExpTuple]):
        for actor_idx, actor_exp in enumerate(experience):
            self.observations[self.ptr, actor_idx, ...] = actor_exp.observation
            self.actions[self.ptr, actor_idx] = actor_exp.action
            self.rewards[self.ptr, actor_idx] = actor_exp.reward
            self.values[self.ptr, actor_idx] = actor_exp.value
            self.log_probs[self.ptr, actor_idx] = actor_exp.log_prob
            self.dones[self.ptr, actor_idx] = float(not actor_exp.done)
        self.ptr += 1

    def add_experiences(self, experiences: List[List[ExpTuple]]):
        assert len(experiences) == self.size + 1
        for experience in experiences[:-1]:
            self.add(experience)
        # experience[-1] for next_values
        for a in range(self.num_agents):
            self.values[-1, a] = experiences[-1][a].value
        self.ptr = 0

    def process_experience(self):
        # compute GAE advantage
        advantages = gae_advantages(self.rewards, self.dones, self.values,
                                    self.gamma, self.lmbda)
        returns = advantages + self.values[:-1, :]

        # concatenate results
        trajectories = (self.observations, self.actions, self.log_probs, returns, advantages)
        trajectories = tuple(map(
            lambda x: np.reshape(x, (self.trajectory_len,) +x.shape[2:]), trajectories))
        return trajectories