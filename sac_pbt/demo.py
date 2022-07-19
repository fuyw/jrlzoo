from typing import Any, Callable, NamedTuple, Tuple
import gym
import jax
import tree
import numpy as np


env_name = "HalfCheetah-v2"
population_size = 10


class TD3HyperParams(NamedTuple):
    lr_policy: float = 3e-4
    lr_critic: float = 3e-4
    discount: float = 0.99
    sigma: float = 0.1
    delay: int = 2
    target_sigma: float = 0.2
    noise_clip: float = 0.5
    tau: float = 0.005


"""
TD3HyperParams(
    lr_policy=array([0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003]),
    lr_critic=array([0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003]),
    discount=array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]),
    sigma=array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    delay=array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
    target_sigma=array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    noise_clip=array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    tau=array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005])
)
"""
hyperparams = tree.map_structure(
    lambda x: np.array([x for _ in range(population_size)]),
    TD3HyperParams())

"""
agent = TD3PBT(env_name=env_name,
               seed=0,
               population_size=population_size,
               hidden_dims=(256, 256))
"""
env = gym.make(env_name)
act_dim = env.action_space.shape[0]


class TD3Networks(NamedTuple):
    policy_network: Callable[Any]
    critic_network: Callable[Any]
    twin_critic_network: Callable[Any]

    init_policy_network: Callable[Any]
    init_critic_network: Callable[Any]
    init_twin_critic_network: Callable[Any]

    add_policy_noise: Callable[Any]



class TD3PBT:
    def __init__(self,
                 env_name: str,
                 seed: int,
                 population_size: int,
                 hidden_dims: Tuple[int]=(256, 256)):

        self.seed = seed
        self.population_size = population_size
        self.random_key = jax.random.PRNGKey(seed)


###################
# Multiple Agents #
###################
state_list = [make_initial_training_state() for _ in range(population_size)]
# initial_training_state = 0
from pbt_utils import make_default_networks
networks = make_default_networks(gym_env=env, hidden_layer_sizes=(256, 256,))

