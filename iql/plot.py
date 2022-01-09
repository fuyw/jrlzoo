import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

env_names = ['halfcheetah', 'hopper', 'walker2d']
levels = ['medium', 'medium-replay', 'medium-expert']
tasks = [f'{env_name}-{level}-v2' for env_name in env_names for level in levels]


def read_data(task_name, seeds):
    res = []
    for seed in seeds:
        res_df = pd.read_csv(f'logs/{task_name}/{seed}.csv', index_col=0)
        rewards = res_df['reward'].values
        res.append(rewards.reshape(-1, 1))
    res = np.concatenate(res, axis=1)
    return res


seeds = range(5)
fix, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))
plt.subplots_adjust(hspace=0.4)
for task_idx, task_name in enumerate(tasks):
    ax = axes[task_idx // 3][task_idx % 3]
    res = read_data(task_name, seeds)
    mu = res.mean(axis=1)
    std = res.std(axis=1)
    ax.plot(range(len(mu)), mu, ls='solid', lw=0.6, label=f'({mu[-10:].mean():.2f})')
    ax.fill_between(range(len(mu)), mu+std, mu-std, alpha=0.3)  #, edgecolor=fill_color, facecolor=fill_color)
    ax.set_title(task_name, fontsize=12)
    ax.legend(fontsize=11)
plt.savefig('a.png')
