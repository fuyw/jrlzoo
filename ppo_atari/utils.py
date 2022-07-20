from typing import List, Tuple
import collections
import logging
import functools
import jax
import jax.numpy as jnp
import optax
import numpy as np

ExpTuple = collections.namedtuple(
    'ExpTuple', ['state', 'action', 'reward', 'value', 'log_prob', 'done'])


def get_logger(fname):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


def process_experience(experience: List[List[ExpTuple]],
                       actor_steps: int,
                       num_agents: int,
                       gamma: float,
                       lmbda: float,
                       obs_shape: Tuple[int] = (84, 84, 4)):
    """Process experiences for Atari agents

    Default: continuous observations & discrete actions.
    """
    observations = np.zeros((actor_steps, num_agents, *obs_shape), dtype=np.float32)
    actions = np.zeros((actor_steps, num_agents), dtype=np.int32)
    rewards = np.zeros((actor_steps, num_agents), dtype=np.float32)
    values = np.zeros((actor_steps + 1, num_agents), dtype=np.float32)
    log_probs = np.zeros((actor_steps, num_agents), dtype=np.float32)
    dones = np.zeros((actor_steps, num_agents), dtype=np.float32)
    assert len(experience) == actor_steps + 1
    for t in range(len(experience) - 1):
        for agent_id, exp_agent in enumerate(experience[t]):
            observations[t, agent_id, ...] = exp_agent.state
            actions[t, agent_id] = exp_agent.action
            rewards[t, agent_id] = exp_agent.reward
            values[t, agent_id] = exp_agent.value
            log_probs[t, agent_id] = exp_agent.log_prob
            # Dones need to be 0 for terminal observations.
            dones[t, agent_id] = float(not exp_agent.done)

    # experience[-1] for next_values
    for a in range(num_agents):
        values[-1, a] = experience[-1][a].value

    # compute GAE advantage
    advantages = gae_advantages(rewards, dones, values, gamma, lmbda)
    targets = advantages + values[:-1, :]

    # concatenate results
    trajectories = (observations, actions, log_probs, targets, advantages)
    trajectory_len = num_agents * actor_steps
    trajectories = tuple(
        map(lambda x: np.reshape(x, (trajectory_len, *x.shape[2:])),
            trajectories))
    return trajectories


@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(rewards: np.ndarray, terminal_masks: np.ndarray,
                   values: np.ndarray, discount: float, gae_param: float):
    """Use GAE to compute advantages.

    As defined by Eq. (11-12) in PPO paper. Implementation uses key observation
    that A_{t} = delta_t + gamma * lmbda * A_{t+1}.
    """
    assert rewards.shape[0] + 1 == values.shape[0], ("One more value needed.")
    advantages = []
    gae = 0.
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal observations.
        value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
        delta = rewards[t] + value_diff

        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + discount * gae_param * terminal_masks[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)


def get_lr_scheduler(config, loop_steps, iterations_per_step):
    # set lr scheduler
    if config.decaying_lr_and_clip_param:
        transition_steps = loop_steps * config.num_epochs * iterations_per_step
        lr = optax.linear_schedule(init_value=config.lr,
                                   end_value=0.,
                                   transition_steps=transition_steps)
    else:
        lr = config.lr
    return lr
