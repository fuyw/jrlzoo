import gym
import d4rl
import numpy as np


env = gym.make('hopper-medium-v2')

# load collected trajectories
traj_data = np.load('saved_buffers/Hopper-v2/5agents.npz')
observations = traj_data['observations']
actions = traj_data['actions']
next_observations = traj_data['next_observations']
rewards = traj_data['rewards']
dones = 1 - traj_data['discounts']


idx = 10
# [ 1.2473539  -0.00170619 -0.00639696 -0.00305187  0.00446615 -0.05353047
#   -0.07389784 -0.54330425 -0.50124    -0.24240645 -0.02062798]
print(f'\nobs = {observations[idx]}') 

# [-0.36295012, -0.23312806, -0.14918306]
print(f'\nact = {actions[idx]}')

# [0.91848462]
print(f'\nrew = {rewards[idx]}')

# [ 1.2473539 , -0.00170619, -0.00639696, -0.00305187,  0.00446615, -0.05353047,
#   -0.07389784, -0.54330425, -0.50124   , -0.24240645, -0.02062798]
print(f'\nnext_obs = {next_observations[idx]}')


_ = env.reset()
print(env.state)
env.state = env.unwrapped.state = observations[300]
print(env.state)
env.step(actions[idx])
print(env.state)


env = gym.make('hopper-medium-v2')
_ = env.reset()
saved_state = env.sim.get_state()
print(saved_state)
saved_state.qpos[1:] = observations[300][:5]
saved_state.qvel = osbervations[300][5:]
# env.sim.set_state(saved_state)
# _ = env.reset()
# print(env.state)
