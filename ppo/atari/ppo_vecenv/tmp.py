import env_utils
import numpy as np

env_name = "PongNoFrameskip-v4"
vec_env = env_utils.create_vec_env(env_name, 5, True, range(5))

o1 = vec_env.reset()  # (5, 84, 84, 4)
actions = np.array([0, 1, 2, 3, 2])
o2, r2, d2, _ = vec_env.step(actions)
