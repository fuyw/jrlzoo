import logging
from collections import deque, namedtuple

import jax
import numpy as np
from flax.core import FrozenDict

Experience = namedtuple("Experience",
                        ["observation", "action", "reward", "done"])
Batch = namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "next_observations", "discounts"])


class ReplayBuffer:
    def __init__(self, max_size, obs_shape=(84, 84), context_len=4):
        self.max_size = int(max_size)
        self.obs_shape = obs_shape
        self.context_len = int(context_len)

        self.obs = np.zeros((self.max_size, ) + obs_shape, dtype='uint8')  # (N, 84, 84)
        self.action = np.zeros((self.max_size, ), dtype='int32')           # (N,)
        self.reward = np.zeros((self.max_size, ), dtype='float32')
        self.done = np.zeros((self.max_size, ), dtype='bool')

        self._curr_size = 0
        self._curr_pos = 0
        self._context = deque(maxlen=context_len - 1)

    def add(self, exp):
        """append a new experience into replay memory
        """
        if self._curr_size < self.max_size:
            self._assign(self._curr_pos, exp)
            self._curr_size += 1
        else:
            self._assign(self._curr_pos, exp)
        self._curr_pos = (self._curr_pos + 1) % self.max_size
        if exp.done:
            self._context.clear()
        else:
            self._context.append(exp)

    def recent_obs(self):
        """ maintain recent obs for training"""
        lst = list(self._context)
        obs = [np.zeros(self.obs_shape, dtype='uint8')] * \
                    (self._context.maxlen - len(lst))
        obs.extend([k.observation for k in lst])
        return obs

    def sample(self, idx):
        """ return obs, action, reward, done,
            note that some frames in obs may be generated from last episode,
            they should be removed from obs
            """
        obs = np.zeros(
            (self.context_len + 1, ) + self.obs_shape, dtype=np.uint8)
        obs_idx = np.arange(idx, idx + self.context_len + 1) % self._curr_size

        # confirm that no frame was generated from last episode
        has_last_episode = False
        for k in range(self.context_len - 2, -1, -1):
            to_check_idx = obs_idx[k]
            if self.done[to_check_idx]:
                has_last_episode = True
                obs_idx = obs_idx[k + 1:]
                obs[k + 1:] = self.obs[obs_idx]
                break

        if not has_last_episode:
            obs = self.obs[obs_idx]

        real_idx = (idx + self.context_len - 1) % self._curr_size
        action = self.action[real_idx]
        reward = self.reward[real_idx]
        done = self.done[real_idx]
        return obs, reward, action, done

    def __len__(self):
        return self._curr_size

    def size(self):
        return self._curr_size

    def _assign(self, pos, exp):
        self.obs[pos] = exp.observation
        self.reward[pos] = exp.reward
        self.action[pos] = exp.action
        self.done[pos] = exp.done

    def sample_batch(self, batch_size):
        """sample a batch from replay memory for training
        """
        batch_idx = np.random.randint(
            self._curr_size - self.context_len - 1, size=batch_size)
        batch_idx = (self._curr_pos + batch_idx) % self._curr_size
        batch_exp = [self.sample(i) for i in batch_idx]
        return self._process_batch(batch_exp)

    def _process_batch(self, batch_exp):
        obs = np.asarray([e[0] for e in batch_exp], dtype='uint8')
        reward = np.asarray([e[1] for e in batch_exp], dtype='float32')
        action = np.asarray([e[2] for e in batch_exp], dtype='int8')
        done = np.asarray([e[3] for e in batch_exp], dtype='bool')

        obs = np.moveaxis(obs, 1, -1)
        return Batch(observations=obs[:, :, :, :self.context_len],
                     actions=action,
                     rewards=reward,
                     next_observations=obs[:, :, :, 1:],
                     discounts=1.-done)

    def save(self, fname):
        np.savez(fname,
                 observations=self.obs,
                 actions=self.action,
                 rewards=self.reward,
                 dones=self.done,
                 curr_size=self._curr_size,
                 curr_pos=self._curr_pos)

    def load(self, fname):
        dataset = np.load(fname)
        self.obs = dataset["observations"]
        self.action = dataset["actions"]
        self.reward = dataset["rewards"]
        self.done = dataset["dones"]
        self._curr_size = dataset["curr_size"]
        self._curr_pos = dataset["curr_pos"]
        self.max_size = len(self.obs)


# Exploration linear decay
def linear_schedule(start_epsilon: float, end_epsilon: float, duration: int,
                    t: int):
    slope = (end_epsilon - start_epsilon) / duration
    return max(slope * t + start_epsilon, end_epsilon)


# Logger
def get_logger(fname):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


def target_update(params: FrozenDict, target_params: FrozenDict,
                  tau: float) -> FrozenDict:

    def _update(param: FrozenDict, target_param: FrozenDict):
        return tau * param + (1 - tau) * target_param

    updated_params = jax.tree_util.tree_map(_update, params, target_params)
    return updated_params
