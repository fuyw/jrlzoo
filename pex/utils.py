import collections
import logging
import os

import gym
import d4rl
import jax
import numpy as np
from flax.core import FrozenDict
from tqdm import trange

Batch = collections.namedtuple("Batch", [
    "observations", "actions", "rewards", "discounts", "next_observations",
    "idx", "flags"
])

SFPBatch = collections.namedtuple("Batch", [
    "observations", "actions", "last_actions", "rewards",
    "discounts", "next_observations", "idx"
])

PERBatch = collections.namedtuple("PER_Batch", [
    "observations", "actions", "rewards", "discounts", "next_observations",
    "idx", "weights", "flags"
])

D4RL_REWARDS = {
    'hopper-random-v2': (2.6980462670326237, 292.5542140603065),
    'hopper-medium-v2': (187.7212032675743, 3218.0389815568924),
    'hopper-medium-replay-v2': (-1.4400692265480757, 3189.348159968853),
    'hopper-medium-expert-v2': (315.8680055141449, 3753.886583685875),
    'halfcheetah-random-v2': (-525.9833031400631, -85.56948105886113),
    'halfcheetah-medium-v2': (-309.7676480333612, 5303.826722159982),
    'halfcheetah-medium-replay-v2': (-636.8851275740599, 4981.297822877765),
    'halfcheetah-medium-expert-v2': (-309.7676480333612, 11239.283746674657),
    'walker2d-random-v2': (-17.00582491233945, 75.03456518426538),
    'walker2d-medium-v2': (-6.605671878904104, 4226.9399804016575),
    'walker2d-medium-replay-v2': (-50.19683612603694, 4126.449885476613),
    'walker2d-medium-expert-v2': (-6.605671878904104, 5006.127595229074)
}


def normalize_reward(env_name, x):
    if "v2" in env_name:
        min_traj_reward, max_traj_reward = D4RL_REWARDS[env_name]
        x = x / (max_traj_reward - min_traj_reward) * 1000.0
    else:
        x -= 1.0
    return x


def load_ckpt(agent, algo, env_name, cnt=200):
    prefix = f"saved_ckpts/{algo}/{env_name}"
    agent.load(f"{prefix}", cnt)


def get_logger(fname: str) -> logging.Logger:
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


def split_into_trajectories(dataset):
    # load d4rl offline dataset
    L = len(dataset["observations"])
    observations = dataset["observations"]
    actions = np.clip(dataset["actions"], -1.0 + 1e-5, 1.0 - 1e-5)
    next_observations = dataset["next_observations"]
    rewards = dataset["rewards"].squeeze()
    terminals = dataset["terminals"].squeeze()

    # mark the end of a trajectory
    dones_float = np.zeros_like(dataset["rewards"])
    for i in range(L-1):
        if np.linalg.norm(
            dataset["observations"][i+1]-dataset["next_observations"][i]
                ) > 1e-6 or dataset["terminals"][i] == 1.0:
            dones_float[i] = 1.0
    dones_float[-1] = 1.0

    trajs = []
    traj_observations = []
    traj_actions = []
    traj_next_observations = []
    traj_rewards = []
    traj_terminals = []
    traj_timeouts = []

    for i in trange(L, desc="[Split into trajs]"):
        traj_observations.append(observations[i])
        traj_actions.append(actions[i])
        traj_next_observations.append(next_observations[i])
        traj_rewards.append(rewards[i])
        traj_terminals.append(terminals[i])
        traj_timeouts.append(False)
        if dones_float[i]==1.0 and i+1<L:
            if traj_terminals[-1] == False:
                traj_timeouts[-1] = True
            trajs.append((np.array(traj_observations),
                          np.array(traj_actions),
                          np.array(traj_next_observations),
                          np.array(traj_terminals),
                          np.array(traj_rewards),
                          np.array(traj_timeouts)))

            traj_observations = []
            traj_actions = []
            traj_next_observations = []
            traj_rewards = []
            traj_terminals = []
            traj_timeouts = []

    return trajs


def filter_trajectories(env_name, pct=0.2):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    trajs = split_into_trajectories(dataset)
    traj_rewards = [(i, traj[-2].sum()) for i, traj in enumerate(trajs)]
    traj_rewards = sorted(traj_rewards, key=lambda x: x[1], reverse=True)

    L = int(len(dataset["rewards"]) * pct)
    current_len = 0

    observations = []
    actions = []
    next_observations = []
    rewards = []
    terminals = []
    i = 0

    while (current_len < L):
        idx = traj_rewards[i][0] 
        observations.append(trajs[idx][0])
        actions.append(trajs[idx][1])
        next_observations.append(trajs[idx][2])
        terminals.append(trajs[idx][3])
        rewards.append(trajs[idx][4])
        current_len += len(trajs[idx][0])
        i += 1

    observations = np.concatenate(observations, axis=0)    
    actions = np.concatenate(actions, axis=0)
    next_observations = np.concatenate(next_observations, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    terminals = np.concatenate(terminals, axis=0)

    os.makedirs(f"saved_buffers/{env_name}", exist_ok=True)
    np.savez(f"saved_buffers/{env_name}/filter{pct}",
             observations=observations,
             actions=actions,
             next_observations=next_observations,
             rewards=rewards,
             terminals=terminals)


def save_traj_data(env_name):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    trajs = split_into_trajectories(dataset)
    observations = []
    actions = []
    next_observations = []
    rewards = []
    terminals = []
    timeouts = []
    for traj in trajs:
        observations.append(traj[0])
        actions.append(traj[1])
        next_observations.append(traj[2])
        terminals.append(traj[3])
        rewards.append(traj[4])
        timeouts.append(traj[5])
    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    next_observations = np.concatenate(next_observations, axis=0)
    terminals = np.concatenate(terminals, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    timeouts = np.concatenate(timeouts, axis=0)
    os.makedirs(f"saved_buffers/{env_name}", exist_ok=True)
    np.savez(f"saved_buffers/{env_name}/trajs",
             observations=observations,
             actions=actions,
             next_observations=next_observations,
             rewards=rewards,
             terminals=terminals,
             timeouts=timeouts)


def get_flow_action_dataset(dataset):
    """ P(a_t | a_{t-1}) condition on one-step action"""
    trajs = split_into_trajectories(dataset)
    actions, next_actions = [], []
    for traj in trajs:
        traj_actions = traj[1]
        actions.append(np.concatenate([
            np.zeros(shape=(1, traj_actions.shape[1])), traj_actions[:-1]], axis=0))
        next_actions.append(traj_actions)
    actions = np.concatenate(actions, axis=0)
    next_actions = np.concatenate(next_actions, axis=0)
    return actions, next_actions
    

######
# ER #
######
class ReplayBuffer:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_size: int = int(1e6),
                 gamma: float = 0.99):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.gamma = gamma

        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim),
                                          dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.flags = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.float32)

    def add(self, observation: np.ndarray, action: np.ndarray,
            next_observation: np.ndarray, reward: float, done: float, flag: bool = False):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.flags[self.ptr] = flag

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=self.observations[idx],
                      actions=self.actions[idx],
                      rewards=self.rewards[idx],
                      discounts=self.discounts[idx] * self.gamma,
                      next_observations=self.next_observations[idx],
                      idx=idx,
                      flags=self.flags[idx])
        return batch

    def convert_D4RL(self, dataset):
        L = len(dataset["observations"])
        print(f"Loaded offline dataset with size {L/1000:.0f}K")
        assert L < self.max_size
        self.observations[:L] = dataset["observations"]
        self.actions[:L] = np.clip(dataset["actions"], -1.0 + 1e-5, 1.0 - 1e-5)
        self.next_observations[:L] = dataset["next_observations"]
        self.rewards[:L] = dataset["rewards"].squeeze()
        self.discounts[:L] = 1. - dataset["terminals"].squeeze()
        self.size = L
        self.ptr = L


class SFPBuffer:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_size: int = int(1e6),
                 gamma: float = 0.99):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.gamma = gamma

        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.last_actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim),
                                          dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.float32)

    def add(self, observation: np.ndarray, action: np.ndarray, last_action: np.ndarray,
            next_observation: np.ndarray, reward: float, done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.last_actions[self.ptr] = last_action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> SFPBatch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = SFPBatch(observations=self.observations[idx],
                         actions=self.actions[idx],
                         last_actions=self.last_actions[idx],
                         rewards=self.rewards[idx],
                         discounts=self.discounts[idx] * self.gamma,
                         next_observations=self.next_observations[idx],
                         idx=idx)
        return batch

    def convert_D4RL(self, dataset):
        L = len(dataset["observations"])
        assert L < self.max_size
        self.observations[:L] = dataset["observations"]
        self.actions[:L] = np.clip(dataset["actions"], -1.0 + 1e-5, 1.0 - 1e-5)
        self.next_observations[:L] = dataset["next_observations"]
        self.rewards[:L] = dataset["rewards"].squeeze()
        self.discounts[:L] = 1. - dataset["terminals"].squeeze()
        self.size = L
        self.ptr = L


############
# Nstep ER #
############
class NstepReplayBuffer:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_size: int = int(1e6),
                 horizon: int = 5,
                 gamma: float = 0.99):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size

        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim),
                                          dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.float32)
        self.flags = np.zeros(max_size, dtype=np.float32)

        self.ptr = 0
        self.size = 0

        # use arrays to simulate queue
        self.queue_observations = np.zeros((horizon, obs_dim),
                                           dtype=np.float32)
        self.queue_actions = np.zeros((horizon, act_dim), dtype=np.float32)
        self.queue_next_observations = np.zeros((horizon, obs_dim),
                                                dtype=np.float32)
        self.queue_rewards = np.zeros(horizon, dtype=np.float32)
        self.queue_dones = np.zeros(horizon, dtype=np.float32)
        self.queue_flags = np.zeros(horizon, dtype=np.float32)
        self.qptr = 0
        self.qsize = 0

        self.horizon = horizon
        self.gamma = gamma
        self.cumulative_discounts = np.array(
            [np.power(gamma, n) for n in range(horizon)])

    def reset_queue(self):
        # reset the queue when we meet a terminal or timeout
        self.queue_observations[:] = 0.
        self.queue_actions[:] = 0.
        self.queue_next_observations[:] = 0.
        self.queue_rewards[:] = 0.
        self.queue_dones[:] = 0.
        self.queue_flags[:] = 0.
        self.qptr = 0
        self.qsize = 0

    def add_queue_to_buffer(self, num=1, done=0.0):
        # indices in temporal sequence
        toadd_indices = np.arange(self.qptr + 1 - self.qsize,
                                  self.qptr + 1) % self.horizon

        # add `num` transitions
        for l in range(num):
            # compute nstep reward
            start_idx = toadd_indices[l]
            queue_indices = toadd_indices[l:]
            queue_rewards = self.queue_rewards[queue_indices]
            queue_len = len(queue_indices)
            nstep_reward = np.sum(queue_rewards *
                                  self.cumulative_discounts[:queue_len])

            # add to the buffer
            self.observations[self.ptr] = self.queue_observations[start_idx]
            self.actions[self.ptr] = self.queue_actions[start_idx]
            self.flags[self.ptr] = self.queue_flags[start_idx]
            self.rewards[self.ptr] = nstep_reward
            self.next_observations[self.ptr] = self.queue_next_observations[
                self.qptr]
            self.discounts[self.ptr] = 1.0 - done

            # update buffer ptr & size
            self.size = min(self.size + 1, self.max_size)
            self.ptr = (self.ptr + 1) % self.max_size

    def add(self,
            observation,
            action,
            next_observation,
            reward,
            done,
            timeout: bool = False,
            flag: float = 0.0):
        # add transition to the queue
        self.queue_observations[self.qptr] = observation
        self.queue_actions[self.qptr] = action
        self.queue_rewards[self.qptr] = reward
        self.queue_next_observations[self.qptr] = next_observation
        self.queue_dones[self.qptr] = done
        self.queue_flags[self.qptr] = flag

        # update queue size
        self.qsize = min(self.qsize + 1, self.horizon)

        # meet timeout transition
        if timeout:
            # add last valid transition when the queue is full
            if self.qsize == self.horizon:
                self.add_queue_to_buffer(num=1, done=0.0)

            # reset the queue
            self.reset_queue()

        # meet normal transition
        else:
            # queue is full
            if self.qsize == self.horizon:
                if done:
                    self.add_queue_to_buffer(num=self.horizon, done=1.0)
                    self.reset_queue()
                else:
                    self.add_queue_to_buffer(num=1, done=0.0)
                    self.qptr = (self.qptr + 1) % self.horizon

            # queue is not full
            else:
                if done:
                    self.add_queue_to_buffer(num=self.qsize, done=1.0)
                    self.reset_queue()
                else:
                    self.qptr = (self.qptr + 1) % self.horizon

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=self.observations[idx],
                      actions=self.actions[idx],
                      rewards=self.rewards[idx],
                      discounts=self.discounts[idx] *
                      np.power(self.gamma, self.horizon),
                      next_observations=self.next_observations[idx],
                      flags=self.flags[idx],
                      idx=idx)
        return batch

    def load_dataset(self, dataset, antmaze=False):
        ds_observations = dataset["observations"]
        ds_actions = dataset["actions"]
        ds_next_observations = dataset["next_observations"]
        ds_rewards = dataset["rewards"]
        if antmaze:
            ds_rewards -= 1.0
        ds_terminals = dataset["terminals"]
        ds_timeouts = dataset["timeouts"]
        for i in range(len(ds_rewards)):
            self.add(observation=ds_observations[i],
                     action=ds_actions[i],
                     next_observation=ds_next_observations[i],
                     reward=ds_rewards[i],
                     done=ds_terminals[i],
                     timeout=ds_timeouts[i])


#######
# PER #
#######
class SumTree:
    """A sum tree data structure for storing replay priorities.

    A sum tree is a complete binary tree whose leaves contain values called
    priorities. Internal nodes maintain the sum of the priorities of all leaf
    nodes in their subtree.

    For max_size = 4, the tree may look like this:

                       +---+
                       |2.5|
                       +-+-+
                         |
                 +-------+--------+
                 |                |
               +-+-+            +-+-+
               |1.5|            |1.0|
               +-+-+            +-+-+
                 |                |
            +----+----+      +----+----+
            |         |      |         |
          +-+-+     +-+-+  +-+-+     +-+-+
          |0.5|     |1.0|  |0.5|     |0.5|
          +---+     +---+  +---+     +---+

    This is stored in a list of numpy arrays:
    self.nodes = [ [2.5], [1.5, 1], [0.5, 1, 0.5, 0.5]  ]

    For conciseness, we allocates arrays as powers of two, and pad the excess
    elements with zero values.

    This is similar to the usual array-based representation of a complete
    binary tree, but it is little more user-friendly.

    """

    def __init__(self, max_size):
        """Create the sum tree data structure.

        Args:
            max_size: int, the maximum number of elements that can be stored
                in this data structure.
        """

        self.levels = [np.zeros(1)]
        level_size = 1
        while level_size < max_size:
            level_size *= 2
            self.levels.append(np.zeros(level_size))

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index
    def sample(self, batch_size):
        """Sample an element from the sum tree.

        Each element has probability p_i / sum_j p_j of being picked, where
        p_i is the (positive) value associated with the node i.

        Args:
            batch_size: int, number of samples in each batch.
        """
        # with stratified sampling
        # bounds = np.linspace(0., self.levels[0][0], batch_size + 1)
        # assert len(bounds) == batch_size + 1
        # segments = [(bounds[i], bounds[i+1]) for i in range(batch_size)]
        # value = [np.random.uniform(x[0], x[1]) for x in segments]

        # without stratified sampling
        value = np.random.uniform(0, self.levels[0][0], size=batch_size)
        ind = np.zeros(batch_size, dtype=int)
        for nodes in self.levels[1:]:
            ind *= 2
            left_sum = nodes[ind]
            is_greater = np.greater(value, left_sum)

            # If value > left_sum -> go right (+1), else go left (+0)
            ind += is_greater

            # IF we go right, we only need to consider the values in the right
            # tree so we substract the sum of values in the left tree
            value -= left_sum * is_greater
        return ind

    def set(self, ind, new_priority):
        """Set the value of a leaf node and update internal nodes accordingly.

        This operation takes O(log(max_size)).
        Args:
            ind: int, the index of the leaf node to be updated.
            new_priority: float, the value which we assign to the node.
        """
        priority_diff = new_priority - self.levels[-1][ind]

        for nodes in reversed(self.levels):
            np.add.at(nodes, ind, priority_diff)
            ind //= 2

    def batch_set(self, ind, new_priority):
        # Confirm we don't increment a node twice
        ind, unique_ind = np.unique(ind, return_index=True)
        priority_diff = new_priority[unique_ind] - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2


class PrioritizedReplayBuffer:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_size: int = int(1e6),
                 anneal_step: int = int(2.5e5),
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 gamma: float = 0.99):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.gamma = gamma

        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim),
                                          dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.float32)
        self.flags = np.zeros(max_size, dtype=np.float32)

        self.tree = SumTree(max_size)
        self.max_priority = 1.0
        self.beta = beta
        self.delta = (
            1 - beta) / anneal_step  # anneal the IS weight with 1e6 steps
        self.alpha = alpha  # add priority ** alpha to the sum_tree

    # add transition with default max_priroity
    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            next_observation: np.ndarray,
            reward: float,
            done: float,
            flag: float = 0.0):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.flags[self.ptr] = flag

        self.tree.set(self.ptr, self.max_priority)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = self.tree.sample(batch_size)
        idx %= self.size

        # importance sampling
        weights = self.tree.levels[-1][idx]**-self.beta
        weights /= weights.max()

        # anneal beta to 1
        self.beta = min(self.beta + self.delta, 1)
        batch = PERBatch(observations=self.observations[idx],
                         actions=self.actions[idx],
                         rewards=self.rewards[idx],
                         discounts=self.discounts[idx] * self.gamma,
                         next_observations=self.next_observations[idx],
                         idx=idx,
                         weights=weights,
                         flags=self.flags[idx])
        return batch

    def update_priority(self, ind, priority):
        # update max priority
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)


class NstepPrioritizedReplayBuffer:

    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_size: int = int(1e6),
                 anneal_step: int = int(2.5e5),
                 alpha: float = 0.3,
                 beta: float = 0.4,
                 gamma: float = 0.99,
                 horizon: int = 5,
                 antmaze: bool = True):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.antmaze = antmaze

        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim),
                                          dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.float32)
        self.flags = np.zeros(max_size, dtype=np.float32)

        self.ptr = 0
        self.size = 0

        # use arrays to simulate queue
        self.queue_observations = np.zeros((horizon, obs_dim),
                                           dtype=np.float32)
        self.queue_actions = np.zeros((horizon, act_dim), dtype=np.float32)
        self.queue_next_observations = np.zeros((horizon, obs_dim),
                                                dtype=np.float32)
        self.queue_rewards = np.zeros(horizon, dtype=np.float32)
        self.queue_dones = np.zeros(horizon, dtype=np.float32)
        self.queue_flags = np.zeros(horizon, dtype=np.float32)
        self.qptr = 0
        self.qsize = 0

        # nstep return
        self.horizon = horizon
        self.gamma = gamma
        self.cumulative_discounts = np.array(
            [np.power(gamma, n) for n in range(horizon)])

        # per
        self.tree = SumTree(max_size)
        self.max_priority = 1.0
        self.beta = beta
        self.delta = (
            1 - beta) / anneal_step  # anneal the IS weight with 1e6 steps
        self.alpha = alpha  # add priority ** alpha to the sum_tree

    def reset_queue(self):
        # reset the queue when we meet a terminal or timeout
        self.queue_observations[:] = 0.
        self.queue_actions[:] = 0.
        self.queue_next_observations[:] = 0.
        self.queue_rewards[:] = 0.
        self.queue_dones[:] = 0.
        self.queue_flags[:] = 0.
        self.qptr = 0
        self.qsize = 0

    def add_queue_to_buffer(self, num=1, done=0.0):
        # indices in temporal sequence
        toadd_indices = np.arange(self.qptr + 1 - self.qsize,
                                  self.qptr + 1) % self.horizon

        # add `num` transitions
        for l in range(num):
            # compute nstep reward
            start_idx = toadd_indices[l]
            queue_indices = toadd_indices[l:]
            queue_rewards = self.queue_rewards[queue_indices]
            queue_len = len(queue_indices)
            nstep_reward = np.sum(queue_rewards *
                                  self.cumulative_discounts[:queue_len])

            # add to the buffer
            self.observations[self.ptr] = self.queue_observations[start_idx]
            self.actions[self.ptr] = self.queue_actions[start_idx]
            self.flags[self.ptr] = self.queue_flags[start_idx]
            self.rewards[self.ptr] = nstep_reward
            self.next_observations[self.ptr] = self.queue_next_observations[self.qptr]
            self.discounts[self.ptr] = 1.0 - done

            # use nstep return 做 reward
            if self.antmaze:
                nstep_priority = np.sum((queue_rewards+1.0) * 
                    self.cumulative_discounts[:queue_len])
            else:
                nstep_priority = nstep_reward
            self.tree.set(self.ptr, np.power(nstep_priority, self.alpha)+1e-3)

            # update per ptr & size
            self.size = min(self.size + 1, self.max_size)
            self.ptr = (self.ptr + 1) % self.max_size

    # add transition with default max_priroity
    def add(self,
            observation: np.ndarray,
            action: np.ndarray,
            next_observation: np.ndarray,
            reward: float,
            done: float,
            timeout: bool = False,
            flag: float = 0.0):
        self.queue_observations[self.qptr] = observation
        self.queue_actions[self.qptr] = action
        self.queue_next_observations[self.qptr] = next_observation
        self.queue_rewards[self.qptr] = reward
        self.queue_dones[self.qptr] = done
        self.queue_flags[self.qptr] = flag

        # update queue size
        self.qsize = min(self.qsize + 1, self.horizon)

        # meet timeout transition
        if timeout:
            # add last valid transition when the queue is full
            if self.qsize == self.horizon:
                self.add_queue_to_buffer(num=1, done=0.0)

            # reset the queue
            self.reset_queue()

        # meet normal transition
        else:
            # queue is full
            if self.qsize == self.horizon:
                if done:
                    self.add_queue_to_buffer(num=self.horizon, done=1.0)
                    self.reset_queue()
                else:
                    self.add_queue_to_buffer(num=1, done=0.0)
                    self.qptr = (self.qptr + 1) % self.horizon

            # queue is not full
            else:
                if done:
                    self.add_queue_to_buffer(num=self.qsize, done=1.0)
                    self.reset_queue()
                else:
                    self.qptr = (self.qptr + 1) % self.horizon

    def sample(self, batch_size):
        idx = self.tree.sample(batch_size)
        idx %= self.size

        # importance sampling
        weights = self.tree.levels[-1][idx]**-self.beta
        weights /= weights.max()

        # anneal beta to 1
        self.beta = min(self.beta + self.delta, 1)
        batch = PERBatch(observations=self.observations[idx],
                         actions=self.actions[idx],
                         rewards=self.rewards[idx],
                         discounts=self.discounts[idx] * np.power(self.gamma, self.horizon),
                         next_observations=self.next_observations[idx],
                         idx=idx,
                         weights=weights.squeeze(),
                         flags=self.flags[idx])
        return batch

    def update_priority(self, ind, priority):
        # update max priority
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)
