from typing import Tuple
import collections
import logging
import jax
import numpy as np

from dm_control import suite
from wrappers import DMC2GYM, wrap_pixels


def make_env(env_name, seed, action_repeat, image_size, num_stack):
    domain_name, task_name = env_name.split('-')
    env = suite.load(domain_name=domain_name,
                     task_name=task_name,
                     task_kwargs={"random": seed})
    env = DMC2GYM(env)

    def wrap(env):
        if "quadruped" in env_name:
            camera_id = 2
        else:
            camera_id = 0
        return wrap_pixels(
            env,
            action_repeat=action_repeat,
            image_size=image_size,
            num_stack=num_stack,
            camera_id=camera_id,
        )

    env = wrap(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def get_logger(fname: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


class EfficientBuffer:
    """
        [s0, s0, s0, s1, s2, s3, ..., s100, s0, s0, s0, s1, s2, ..., s90]
        [ 0,  0,  0,  1,  1,  1, ...,    1,  0,  0,  0,  1,  1, ...,   1]

        [s88, s89, s90, s91, ..., s100, s0, s1, s2, ..., s87, s88, 89, s90]
        [  0,   0,   0,   1, ...,    1,  0,  0,  0, ...,   1,   1,  1,   1]
    """
    def __init__(self,
                 obs_shape: Tuple[int] = (64, 64, 3, 3),
                 act_dim: int = 4,
                 batch_size: int = 256,
                 max_size: int = 100000):
        self.obs_shape = obs_shape  # (64, 64, 3, 3)
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.traj_start = True
        self.batch_size = batch_size
        self.frame_stack = obs_shape[-1]
        self.obs_idx = np.vstack([np.arange(-self.frame_stack, 1)]*batch_size)

        # (100000, 64, 64, 3)
        self.observations = np.zeros((max_size, *obs_shape[:-1]), dtype=np.uint8)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.uint8)
        self.valid_idx = np.zeros(max_size, dtype=bool)

    def add(self,
            observation: np.ndarray,        # (64, 64, 3, 3)
            action: np.ndarray,             # (4,)
            next_observation: np.ndarray,   # (64, 64, 3, 3)
            reward: float,
            done: int,
            terminate: int):

        # when the buffer is full, special treatment for the beginning
        if (self.ptr==0) and (self.size==self.max_size) and (not self.traj_start):
            for idx in range(self.max_size-self.frame_stack, self.max_size):
                # copy frame_stack transitions to the beginning
                tmp_obs = self.observations[idx]
                tmp_action = self.actions[idx]
                tmp_reward = self.rewards[idx]
                tmp_discount = self.discounts[idx]

                self.observations[self.ptr] = tmp_obs
                self.actions[self.ptr] = tmp_action
                self.rewards[self.ptr] = tmp_reward
                self.discounts[self.ptr] = tmp_discount
                self.valid_idx[self.ptr] = False

                self.ptr = (self.ptr + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)

        # beginning of a trajectory
        if self.traj_start:
            for i in range(self.frame_stack):
                self.observations[self.ptr] = observation[..., i]
                self.valid_idx[self.ptr] = False
                self.ptr = (self.ptr + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)

        # use next_observation's last frame
        self.observations[self.ptr] = next_observation[..., -1]
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.valid_idx[self.ptr] = True
        self.traj_start = terminate > 0
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # overwrite next frame_stack
        for i in range(self.frame_stack):
            idx = (self.ptr + 1) % self.max_size
            self.valid_idx[idx] = False

    def sample(self):
        idx = np.random.randint(self.size, size=self.batch_size)
        for i in range(self.batch_size):
            while not self.valid_idx[idx[i]]:
                idx[i] = np.random.randint(self.size)

        total_obs_idx = self.obs_idx + idx.reshape(-1, 1)
        total_observations = self.observations[total_obs_idx.reshape(-1)]
        total_observations = total_observations.reshape(
            self.batch_size, self.frame_stack+1, *self.obs_shape[:-1])
        total_observations = np.moveaxis(total_observations, 1, -1)
        batch = Batch(observations=total_observations,
                      actions=self.actions[idx],
                      rewards=self.rewards[idx],
                      discounts=self.discounts[idx],
                      next_observations=None)
        return batch

    def get_iterator(self, queue_size: int = 2):
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                batch = self.sample()
                queue.append(jax.device_put(batch))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
