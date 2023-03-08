"""D4RL Maze env 
    # sparse reward with 0.0/1.0
    maze2d-umaze-v1
    maze2d-medium-v1
    maze2d-large-v1

    # dense reward with negative exponentiated distance
    maze2d-umaze-dense-v1
    maze2d-medium-dense-v1
    maze2d-large-dense-v1
"""
import os

import d4rl
import gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils import wrap_mazeenv


# plot snapshot by the RL agent
def plot_snapshot_random_agent():
    mode = "rgb_array"  # human, rgb_array
    env_name = "antmaze-large-play-v0"  #"antmaze-large-play-v0"
    os.makedirs("imgs", exist_ok=True)

    # wrap_antenv(env.env._wrapped_env, 5, 4, 1.3)
    env = gym.make(env_name)
    wrap_mazeenv(env, x=1, y=1, z=1.2)
    env.reset()
    for t in range(100):
        action = env.action_space.sample()
        s, _, _, _ = env.step(action)
        if (t + 1) % 10 == 0:
            frame = env.render(mode=mode)
            img = Image.fromarray(np.flipud(frame), "RGB")
            img.save(os.path.join("./imgs", f"{t+1}.png"))


# plot snapshot in D4RL dataset
def plot_snapshot_d4rl_dataset():
    ds = env.get_dataset()
    goals = ds["infos/goal"]
    qpos = ds["infos/qpos"]
    qvel = ds["infos/qvel"]
    actions = ds["actions"]

    t = 0
    env.set_state(qpos[t], qvel[t])
    env.step(actions[t])
    env.set_target(goals[t])
    frame = env.render(mode=mode, width=500, height=500)
    img = Image.fromarray(np.flipud(frame), "RGB")
    img.save(os.path.join("./imgs", f"x.png"))


def plot_trajectory_random_agent(env_name="maze2d-large-v1",
                                 mode="rgb_array",
                                 x=1,
                                 y=1,
                                 z=1.2):
    env = gym.make(env_name)
    wrap_mazeenv(env, x=x, y=y, z=z)
    env.reset()

    frames = []
    for t in range(500):
        action = env.action_space.sample()
        _ = env.step(action)
        if (t + 1) % 10 == 0:
            frame = env.render(mode=mode)
            frames.append(frame)
            # img = Image.fromarray(np.flipud(frame), "RGB")
            # img.save(os.path.join("./imgs", f"{t+1}.png"))
    return frames


frames = plot_trajectory_random_agent()
x = sum(frames[5:]) / (len(frames) - 5)
img = Image.fromarray(np.flipud(x), "RGB")
img.save(os.path.join("./imgs", "background.png"))
# x = frames[0]
# deltas = []
# for i in range(1, len(frames)):
#     delta_frame = frames[i] - frames[i-1]
#     deltas.append(delta_frame)
