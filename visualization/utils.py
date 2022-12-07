import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def wrap_antenv(env, x, y, z=1.2, mode="rgb_array"):
    viewer = env._get_viewer(mode)
    viewer.cam.distance = env.model.stat.extent * z
    viewer.cam.lookat[0] += x
    viewer.cam.lookat[1] += y
    viewer.cam.elevation = -90


def wrap_mazeenv(env, x, y, z=1.2, mode="rgb_array"):
    viewer = env.env._get_viewer(mode)
    viewer.cam.distance = env.model.stat.extent * z
    viewer.cam.lookat[0] += x
    viewer.cam.lookat[1] += y
    viewer.cam.elevation = -90
