import gym
from gym.envs.registration import registry, register

import numpy as np
from utils import ReplayBuffer


def register_custom_envs():
    if "PointmassEasy-v0" not in registry.env_specs:
        register(
            id="PointmassEasy-v0",
            entry_point="envs.pointmass.pointmass:Pointmass",
            kwargs={"difficulty": 0}
        )
    if "PointmassMedium-v0" not in registry.env_specs:
        register(
            id="PointmassMedium-v0",
            entry_point="envs.pointmass.pointmass:Pointmass",
            kwargs={"difficulty": 1}
        )
    if "PointmassHard-v0" not in registry.env_specs:
        register(
            id="PointmassHard-v0",
            entry_point="envs.pointmass.pointmass:Pointmass",
            kwargs={"difficulty": 2}
        )
    if "PointmassVeryHard-v0" not in registry.env_specs:
        register(
            id="PointmassVeryHard-v0",
            entry_point="envs.pointmass.pointmass:Pointmass",
            kwargs={"difficulty": 3}
        )


# register environments
register_custom_envs()


EP_LENS = {
    "PointmassEasy-v0": 50,
    "PointmassMedium-v0": 150,
    "PointmassHard-v0": 100,
    "PointmassVeryHard-v0": 200,
}


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="PointmassHard-v0", choices=(
        "PointmassEasy-v0", "PointmassMedium-v0", "PointmassHard-v0", "PointmassVeryHard-v0"))
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_timesteps", type=int, default=50_000)
    parser.add_argument("--learning_starts", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=0.2)
    args = parser.parse_args()

    # set episode length
    args.ep_len = EP_LENS[args.env_name]
    return args

args = get_args()

# create envs
env = gym.make(args.env_name)
eval_env = gym.make(args.env_name)
env.seed(10)
eval_env.seed(20)

# params
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

action = env.action_space.sample()
next_obs, reward, done, info = env.step(action)

