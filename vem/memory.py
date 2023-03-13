import collections
import numpy as np
from utils import *

Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


class EpisodicMemory:
    def __init__(self, obs_dim, act_dim, max_size):
        self.max_size = max_size

        self.pointer = 0
        # self.ptr = 0
        self.size = 0

        self.capacity = max_size
        self.curr_capacity = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)
        self.q_values = np.zeros(max_size)
        self.states = np.zeros((max_size, obs_dim))

    def add(self, obs, action, state, sampled_return, next_id=-1):

        index = self.pointer
        self.pointer = (self.pointer + 1) % self.capacity

        if self.curr_capacity >= self.capacity:
            # Clean up old entry
            if index in self.end_points:
                self.end_points.remove(index)
            self.prev_id[index] = []
            self.next_id[index] = -1
            self.q_values[index] = -np.inf
        else:
            self.curr_capacity = min(self.capacity, self.curr_capacity + 1)

        # Store new entry
        self.replay_buffer[index] = obs
        self.action_buffer[index] = action
        if state is not None:
            self.query_buffer[index] = state
        self.q_values[index] = sampled_return
        self.returns[index] = sampled_return
        self.lru[index] = self.time

        self._it_sum[index] = self._max_priority ** self._alpha
        self._it_min[index] = self._max_priority ** self._alpha
        if next_id >= 0:
            self.next_id[index] = next_id
            if index not in self.prev_id[next_id]:
                self.prev_id[next_id].append(index)
        self.time += 0.01

        return index
    def retrieve_trajectories(self):
        trajs = []
        for end_point in self.end_points:
            traj = []
            prev = end_point
            while prev is not None:
                traj.append(prev)
                try:
                    prev = self.prev_id[prev][0]
                except IndexError as e:
                    prev = None
            trajs.append(np.array(traj))
        return trajs

    def compute_approximate_return_double(self, observations, actions=None):
        pass

    def update_sequence_with_qs(self, sequence):
        next_id = -1
        Rtd = 0

        for obs, a, z, q_t, r, truly_done, done in reversed(sequence):
            if truly_done:
                Rtd = r
            else:
                Rtd = self.gamma * Rtd + r

            current_id = self.add(obs, a, z, Rtd, next_id)
