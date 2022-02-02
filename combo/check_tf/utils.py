from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow.compat.v1 as tf

import collections
import jax
import numpy as np


def get_required_argument(dotmap, key, message, default=None):
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]],
                        initializer=tf.random_normal_initializer(),
                        trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
       power iteration
       Usually iteration = 1 will be enough
       """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


class TensorStandardScaler:
    """Helper class for automatically normalizing inputs into the network.
    """

    def __init__(self, x_dim):
        """Initializes a scaler.

        Arguments:
        x_dim (int): The dimensionality of the inputs into the scaler.

        Returns: None.
        """
        self.fitted = False
        with tf.variable_scope("Scaler"):
            self.mu = tf.get_variable(name="scaler_mu",
                                      shape=[1, x_dim],
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=False)
            self.sigma = tf.get_variable(
                name="scaler_std",
                shape=[1, x_dim],
                initializer=tf.constant_initializer(1.0),
                trainable=False)

        self.cached_mu, self.cached_sigma = np.zeros([0, x_dim
                                                      ]), np.ones([1, x_dim])

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        if len(data.shape) == 3:
            mu = np.mean(np.reshape(data, [-1, data.shape[-1]]),
                         axis=0,
                         keepdims=True)
            sigma = np.std(np.reshape(data, [-1, data.shape[-1]]),
                           axis=0,
                           keepdims=True)
        else:
            mu = np.mean(data, axis=0, keepdims=True)
            sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.mu.load(mu)
        self.sigma.load(sigma)
        self.fitted = True
        self.cache()

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.sigma

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.sigma * data + self.mu

    def get_vars(self):
        """Returns a list of variables managed by this object.

        Returns: (list<tf.Variable>) The list of variables.
        """
        return [self.mu, self.sigma]

    def cache(self):
        """Caches current values of this scaler.

        Returns: None.
        """
        self.cached_mu = self.mu.eval()
        self.cached_sigma = self.sigma.eval()

    def load_cache(self):
        """Loads values from the cache

        Returns: None.
        """
        self.mu.load(self.cached_mu)
        self.sigma.load(self.cached_sigma)


Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


QBatch = collections.namedtuple(
    "QBatch",
    ["observations", "actions", "rewards", "discounts", "next_observations", "qpos", "qvel"])


def get_training_data(replay_buffer, ensemble_num=7, holdout_num=1000):
    """
    inputs.shape  = (N, obs_dim+act_dim)
    targets.shape = (N, obs_dim+1) 
    holdout_inputs.shape  = (E, N, obs_dim+act_dim)
    holdout_targets.shape = (E, N, obs_dim+1)
    """
    # load the offline data
    observations = replay_buffer.observations
    actions = replay_buffer.actions
    next_observations = replay_buffer.next_observations
    rewards = replay_buffer.rewards.reshape(-1, 1)  # reshape for correct shape
    delta_observations = next_observations - observations

    # prepare for model inputs & outputs
    inputs = np.concatenate([observations, actions], axis=-1)
    targets = np.concatenate([rewards, delta_observations], axis=-1)

    # validation dataset
    permutation = np.random.permutation(inputs.shape[0])

    # split the dataset
    inputs, holdout_inputs = inputs[permutation[holdout_num:]], inputs[permutation[:holdout_num]]
    targets, holdout_targets = targets[permutation[holdout_num:]], targets[permutation[:holdout_num]]
    holdout_inputs = np.tile(holdout_inputs[None], [ensemble_num, 1, 1])
    holdout_targets = np.tile(holdout_targets[None], [ensemble_num, 1, 1])

    return inputs, targets, holdout_inputs, holdout_targets


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)

    def add(self, observation: np.ndarray, action: np.ndarray,
            next_observation: np.ndarray, reward: float, done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def add_batch(self, observations, actions, next_observations, rewards, dones):
        add_num = len(actions)
        add_idx = np.arange(self.ptr, self.ptr + add_num) % self.max_size
        self.observations[add_idx] = observations
        self.actions[add_idx] = actions
        self.next_observations[add_idx] = next_observations
        self.rewards[add_idx] = rewards.reshape(-1)
        self.discounts[add_idx] = (1 - dones).reshape(-1)
        self.ptr = (self.ptr + add_num) % self.max_size
        self.size = min(self.size + add_num, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=jax.device_put(self.observations[idx]),
                      actions=jax.device_put(self.actions[idx]),
                      rewards=jax.device_put(self.rewards[idx]),
                      discounts=jax.device_put(self.discounts[idx]),
                      next_observations=jax.device_put(
                          self.next_observations[idx]))
        return batch

    def convert_D4RL(self, dataset):
        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.next_observations = dataset["next_observations"]
        self.rewards = (dataset["rewards"] - 0.5) * 4.0
        self.discounts = 1. - dataset["terminals"]
        self.size = self.observations.shape[0]
    
    def convert_D4RL2(self, dataset):
        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.next_observations = dataset["next_observations"]
        self.rewards = dataset["rewards"]
        self.discounts = 1. - dataset["terminals"]
        self.size = self.observations.shape[0]
        self.qpos = dataset["qpos"]
        self.qvel = dataset["qvel"]

    def normalize_states(self, eps: float = 1e-3):
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean)/std
        self.next_observations = (self.next_observations - mean)/std
        return mean, std


class InfoBuffer:
    def __init__(self, obs_dim: int, act_dim: int, qpos_dim: int = None,
                 qvel_dim: int = None, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)
        self.qpos = np.zeros((max_size, qpos_dim))
        self.qvel = np.zeros((max_size, qvel_dim))

    def add(self, observation: np.ndarray, action: np.ndarray,
            next_observation: np.ndarray, reward: float, done: float,
            qpos: np.ndarray, qvel: np.ndarray):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done
        self.qpos[self.ptr] = qpos
        self.qvel[self.ptr] = qvel

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = QBatch(observations=jax.device_put(self.observations[idx]),
                       actions=jax.device_put(self.actions[idx]),
                       rewards=jax.device_put(self.rewards[idx]),
                       discounts=jax.device_put(self.discounts[idx]),
                       next_observations=jax.device_put(self.next_observations[idx]),
                       qpos=self.qpos[idx],
                       qvel=self.qvel[idx])
        return batch

    def save(self, fname):
        np.savez(fname,
                 observations=self.observations[:self.size],
                 actions=self.actions[:self.size],
                 next_observations=self.next_observations[:self.size],
                 rewards=self.rewards[:self.size],
                 discounts=self.discounts[:self.size],
                 qpos=self.qpos[:self.size],
                 qvel=self.qvel[:self.size])

    def load(self, fname='saved_buffers/Hopper-v2/s0.npz'):
        dataset = np.load(fname)
        self.observations = dataset['observations']
        self.actions = dataset['actions']
        self.next_observations = dataset['next_observations']
        self.rewards = dataset['rewards'].reshape(-1)
        self.discounts = dataset['discounts'].reshape(-1)
        self.qpos = dataset['qpos']
        self.qvel = dataset['qvel']
        self.size = len(self.actions)


def check_replay_buffer():
    import gym
    task = 'Hopper'
    env = gym.make(f'{task}-v2')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env_state = env.sim.get_state()
    qpos_dim = len(env_state.qpos)
    qvel_dim = len(env_state.qvel)
    replay_buffer = InfoBuffer(obs_dim, act_dim, qpos_dim, qvel_dim)
    replay_buffer.load(f'saved_buffers/{task}-v2/s0.npz')

    L = replay_buffer.size
    _ = env.reset()
    env_state = env.sim.get_state()
    error_lst = []
    for i in trange(100):
        env_state.qpos[:] = replay_buffer.qpos[i]
        env_state.qvel[:] = replay_buffer.qvel[i]
        env.sim.set_state(env_state)
        act = replay_buffer.actions[i]
        next_obs, reward, done, _ = env.step(act)
        obs_error = abs(replay_buffer.next_observations[i] - next_obs).sum()
        rew_error = abs(replay_buffer.rewards[i] - reward)
        print((i, obs_error, rew_error))
        # if np.allclose(replay_buffer.next_observations[i], next_obs) and np.allclose(replay_buffer.rewards[i], reward):
        #     pass
        # else:
        #     error_lst.append(i)
