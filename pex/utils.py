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
    agent.load(prefix, cnt)


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
