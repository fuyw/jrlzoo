import os
import gym
import d4rl
import numpy as np
import jax.numpy as jnp
from models import DynamicsModel
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

env = gym.make('hopper-medium-v2')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
ensemble_num = 7
model = DynamicsModel(env = 'hopper-medium-v2', ensemble_num=ensemble_num)

# agent.model.train()
model.load('ensemble_models/hopper-medium-v2/s42')


observations = np.random.normal(size=(16, obs_dim))
actions = np.random.normal(size=(16, act_dim))

x = jnp.concatenate([observations, actions], axis=-1)  # (16, 14)
model_mu, model_log_var = model.model.apply({"params": model.model_state.params}, x)

#########
# Check #
#########
x0 = x[0].reshape(1, -1)  # (1, 14)
x1 = jnp.repeat(jnp.expand_dims(x[0], axis=0), repeats=7, axis=0)  # (7, 14)

model_mu, model_log_var = model.model.apply({"params": model.model_state.params}, x0)
model_mu1, model_log_var1 = model.model.apply({"params": model.model_state.params}, x1)

model_mu - model_mu1
