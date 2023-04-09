import collections
import envpool
import jax
import jax.numpy as jnp
import logging
import numpy as np
from flax.core import FrozenDict


###################
# Utils Functions #
###################
def get_logger(fname):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


def target_update(params: FrozenDict, target_params: FrozenDict, tau: float) -> FrozenDict:
    def _update(param: FrozenDict, target_param: FrozenDict):
        return tau*param + (1-tau)*target_param
    updated_params = jax.tree_util.tree_map(_update, params, target_params)
    return updated_params


def get_kernel_norm(kernel_params: jnp.array):
    return jnp.linalg.norm(kernel_params)


#################
# Replay Buffer #
#################
Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.discounts = np.zeros(max_size, dtype=np.float32)

    def add(self, observation: np.ndarray, action: np.ndarray,
            next_observation: np.ndarray, reward: float, done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=self.observations[idx],
                      actions=self.actions[idx],
                      rewards=self.rewards[idx],
                      discounts=self.discounts[idx],
                      next_observations=self.next_observations[idx])
        return batch

    def convert_D4RL(self, dataset):
        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.next_observations = dataset["next_observations"]
        self.rewards = dataset["rewards"].squeeze()
        self.discounts = 1. - dataset["terminals"].squeeze()
        self.size = self.observations.shape[0]

    def normalize_states(self, eps: float = 1e-3):
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean)/std
        self.next_observations = (self.next_observations - mean)/std
        return mean, std


###########################
# Neural Episodic Control #
###########################
class Grid:
    def __init__(self,
                 min_val: np.ndarray,
                 max_val: np.ndarray,
                 grid_num: int,
                 clipped: bool = True):

        self.min_val = min_val
        self.max_val = max_val
        self.k = grid_num
        self.dim = min_val.shape[0]
        self.state_num = np.power(grid_num, self.dim)
        self.delta = (max_val - min_val) / self.k
        self.clipped = clipped

    def state_abstract(self, raw_states):
        abstract_states = np.zeros(raw_states.shape[0], dtype=np.int32)


class ScoreInspector:
    def __init__(self,
                 step: int,
                 grid_num: int,
                 obs_dim: int,
                 hid_dim: int,
                 act_dim: int,
                 max_obs: float,
                 max_action: float,
                 reduction: bool = False):

        self.step = step
        self.grid_num = grid_num
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim
        self.max_obs = max_obs
        self.max_action = max_action
        self.reduction = reduction

        # (state, action) pair
        self.state_dim = obs_dim + act_dim

        self.setup()

    def setup(self):
        if self.reduction:
            self.project_matrix = np.random.uniform(0, 0.1, (self.state_dim, self.hid_dim))
            self.min_state = np.dot(np.array([-self.max_obs for _ in range(self.obs_dim)] + [
                -self.max_action for _ in range(self.act_dim)]), self.project_matrix)
            self.max_state = np.dot(np.array([self.max_obs for _ in range(self.obs_dim)] + [
                self.max_action for _ in range(self.act_dim)]), self.project_matrix)
        else:
            self.min_state = np.array([-self.max_obs for _ in range(self.obs_dim)] + [
                -self.max_action for _ in range(self.act_dim)])
            self.max_state = np.array([self.max_obs for _ in range(self.obs_dim)] + [
                self.max_action for _ in range(self.act_dim)])

        self.min_avg_proceed = 0
        self.max_avg_proceed = 1000

        self.avg_score = 0

        self.states_info = dict()

        self.grid = Grid(self.min_state, self.max_state, self.grid_num)


class Abstractor:
    def __init__(self, step, epsilon):
        self.step = step
        self.epsilon = epsilon
        self.inspector = None

    def dim_reduction(self, state):
        small_state = np.dot(state, self.inspector.project_matrix)


class NECReplayBuffer:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_obs: float,
                 max_action: float):

        self.max_obs = max_obs
        self.max_action = max_action

        self.obs_list = []
        self.obs_action_list = []
        self.reward_list = []
        self.episode_reward = []
        