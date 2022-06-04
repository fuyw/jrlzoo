from typing import Any, Callable
import multiprocessing
import functools

import jax
import flax
import numpy as np

import env_utils


@functools.partial(jax.jit, static_argnums=0)
def policy_action(apply_fn: Callable[..., Any],
                  params: flax.core.frozen_dict.FrozenDict,
                  state: np.ndarray):
    """Forward pass of the network.
    
    Args:
        apply_fn: forward pass of the actor-critic model
        params: parameters of the actor-critic model
        state: model inputs
    
    Returns:
        out: a tuple (log_probabilities, values)
    """
    out = apply_fn({"params": params}, state)
    return out


class RemoteSimulator:
    """Emulating Atari in a separate process.
    An object of this class is created for every agent.
    """
    def __init__(self, game: str):
        """Start the remote process and create Pipe() to communicate with it."""
        parent_conn, child_conn = multiprocessing.Pipe()
        self.proc = multiprocessing.Process(
            target=rcv_action_send_exp, args=(child_conn, game))
        self.proc.daemon = True
        self.conn = parent_conn
        self.proc.start()


def rcv_action_send_exp(conn, game: str):
    """Run the remote agents.

    Receive action from the main learner, perform one step of simulation and
    send back collected experience.
    """
    env = env_utils.create_env(game, clip_rewards=True)
    while True:
        obs, done = env.reset(), False

        # Observations fetched from Atari env need additional batch
        state = obs[None, ...]
        while not done:
            conn.send(state)
            action = conn.recv()
            obs, reward, done, _ = env.step(action)
            next_state = obs[None, ...] if not done else None
            experience = (state, action, reward, done)
            conn.send(experience)
            if done: break
            state = next_state
