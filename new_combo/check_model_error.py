import gym, d4rl
import numpy as np
import pandas as pd
from load_tf_model import load_model


FDIR = '/usr/local/data/yuweifu/jaxrl/td3/saved_buffers'


res = []
for task in ['HalfCheetah', 'Walker2d', 'Hopper']:
    buffer_dir = f'{FDIR}/{task}-v2/5agents.npz'
    dataset = np.load(buffer_dir)

    real_obs = dataset['observations']
    real_act = dataset['actions']
    real_next_obs = dataset['next_observations']
    real_rew = dataset['rewards']
    real_discounts = dataset['discounts']

    for level in ['medium', 'medium-replay', 'medium-expert']:
        env_name = f'{task.lower()}-{level}-v2'
        try:
            model = load_model(env_name[:-3])
        except Exception as e:
            continue

        batch_num = np.ceil(len(real_obs) / 10000)
        model_next_obs, model_rew, model_done = [], [], []

        for i in range(int(batch_num)):
            batch_obs = real_obs[i*10000: (i+1)*10000]
            batch_act = real_act[i*10000: (i+1)*10000]        
            batch_model_next_obs, batch_model_rew, batch_model_done, _ = model.step(
                batch_obs, batch_act)        
            model_next_obs.append(batch_model_next_obs)
            model_rew.append(batch_model_rew)
            model_done.append(batch_model_done)

        model_next_obs = np.concatenate(model_next_obs)
        model_rew = np.concatenate(model_rew).squeeze()
        model_done = np.concatenate(model_done)

        model_rew = model_rew / 4.0 + 0.5
        model_discounts = 1 - model_done.squeeze()
        assert real_next_obs.shape == model_next_obs.shape
        assert real_rew.shape == model_rew.shape
        # assert real_discounts.shape == model_discounts.shape
        obs_error = abs(real_next_obs - model_next_obs).mean()
        rew_error = abs(real_rew - model_rew).mean()

        res.append((env_name, obs_error, rew_error))

res_df = pd.DataFrame(res, columns=['env', 'obs_error', 'rew_error'])
res_df.to_csv('model_error.csv')
