import gym
import jax
import pandas as pd
import matplotlib.pyplot as plt
from models import DynamicsModel
from utils import get_training_data

seed = 42
elite_num = 5
ensemble_num = 7
env_name = 'hopper-medium-v2'

env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Load pre-trained model
model = DynamicsModel(env_name, seed, ensemble_num, elite_num)
inputs, targets, holdout_inputs, holdout_targets = get_training_data(
    model.replay_buffer, ensemble_num, model.holdout_num)


_, ax = plt.subplots()
wds = [0.01, 0.05, 0.001, 0.005, 0.0001, 1e-5, 3e-5]
for wd in wds:
    df = pd.read_csv(f'ensemble_models/hopper-medium-v2/s42_wd{wd}.csv', index_col=0)
    ax.plot(range(len(df)), df['val_loss'].values, label=f'{wd}')
    ax.legend()
plt.savefig('val_losses.png')

# model.load('ensemble_models/hopper-medium-v2/s42_wd0.01')


