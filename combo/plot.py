import pandas as pd
import matplotlib.pyplot as plt

import gym, d4rl
env_name = 'hopper-medium-v2'
env = gym.make(env_name)
# env.get_normalized_score(avg_reward) * 100

cols = ['evaluation/return-average', 'Q_loss']
res_df = pd.read_csv(f'logs/{env_name}/progress.csv')
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
for i in range(2):
    col = cols[i]
    ax = axes[i]
    data = res_df[col].rolling(window=10, min_periods=1).mean()
    if i == 0: data = [env.get_normalized_score(i)*100 for i in data]
    ax.plot(res_df['train-steps'], data)
    if col == 'Q_loss':
        ax.set_yscale('log')
    ax.set_title(col)
plt.savefig('baseline_hopper.png')
plt.close()

