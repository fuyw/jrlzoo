import envpool
import numpy as np

env = envpool.make_gym("Pong-v5", num_envs=5)   
env.reset()
action = np.array([0, 1, 2, 1, 0])
obs, rew, done, info = env.step(action)
