import os
import gym
import sys
import train
from absl import app, flags
from ml_collections import config_flags
from atari_wrappers import wrap_deepmind

config_flags.DEFINE_config_file("config", default="configs/atari.py")
FLAGS = flags.FLAGS
FLAGS(sys.argv)
config = FLAGS.config


env = gym.make(f"{config.env_name}NoFrameskip-v4")
env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")
eval_env = gym.make(f"{config.env_name}NoFrameskip-v4")
eval_env = wrap_deepmind(eval_env, dim=84, obs_format="NHWC", test=True)
act_dim = env.action_space.n



def compare_torch_model():
    import torch
    import torch.nn as nn
    obs = torch.randn(1, 4, 84, 84)
    conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5,
                      stride=1, padding=2)
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    x = conv1(obs)  # [1, 32, 84, 84]
    x = max_pool(x)  # [1, 32, 42, 42]
    print(x.shape)  


def compare_jax_model():
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"
    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    class QNetwork(nn.Module):
        act_dim: int

        def setup(self):
            self.conv1 = nn.Conv(features=32, kernel_size=(5, 5), strides=(1, 1),
                                padding=(2, 2), name="conv1")
            self.conv2 = nn.Conv(features=32, kernel_size=(5, 5), strides=(1, 1),
                                padding=(2, 2), name="conv2")
            self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                                padding=(1, 1), name="conv3")
            self.conv4 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                                padding=(1, 1), name="conv4")
            self.fc_layer = nn.Dense(features=512, name="fc")
            self.out_layer = nn.Dense(features=self.act_dim, name="out")
            # self.max_pool = nn.max_pool()

        def __call__(self, observation):
            x = observation.astype(jnp.float32) / 255.               # (1, 84, 84, 32)
            x = nn.relu(self.conv1(x))
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (1, 42, 42, 32)
            x = nn.relu(self.conv2(x))
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (1, 21, 21, 32)
            x = nn.relu(self.conv3(x))
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (1, 10, 10, 64)
            x = x.reshape(len(observation), -1)                      # (6400,)
            x = nn.relu(self.fc_layer(x))                            # (512,)
            Qs = self.out_layer(x)                                   # (act_dim,)
            return Qs


    rng = jax.random.PRNGKey(0)
    q_network = QNetwork(4)
    obs = jnp.ones(shape=(1, 84, 84, 4))
    params = q_network.init(rng, obs)["params"]
    x = q_network.apply({"params": params}, obs)
    print(x.shape)  # (1, 4)
