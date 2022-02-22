import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('imgs', exist_ok=True)
colors = [
    "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#8c564b",
    "#e377c2", "#bcbd22", "#17becf"
]
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']


env_names = ['halfcheetah', 'hopper', 'walker2d']
levels = ['medium', 'medium-replay', 'medium-expert']
tasks = [
    f'{env_name}-{level}-v2' for env_name in env_names for level in levels
]


def smooth(x, window=3):
    y = np.ones(window)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x.reshape(-1, 1)


def read_data(log_dir, env_name='hopper-medium-v2', col='reward', window=1):
    rewards = []
    res_lst = []
    csv_files = [i for i in os.listdir(f'{log_dir}/{env_name}') if '.csv' in i]
    for csv_file in csv_files:
        df = pd.read_csv(f'{log_dir}/{env_name}/{csv_file}', index_col=0).set_index('step')
        plot_idx = [10000 * i for i in range(101)]
        res_idx = range(955000, 1005000, 5000)
        x = df.loc[plot_idx, col].values
        res_lst.append(smooth(x, window=window))
        rewards.append(df.loc[res_idx, 'reward'].mean())
    res_lst = np.concatenate(res_lst, axis=-1)
    rewards = np.array(rewards)
    return res_lst, rewards


def plot_ax(ax, data, fill_color, title=None, log=False, label=None):
    sigma = np.std(data, axis=1)
    mu = np.mean(data, axis=1)
    if label:
        ax.plot(range(len(mu)),
                mu,
                color=fill_color,
                ls='solid',
                lw=0.6,
                label=label)
    else:
        ax.plot(range(len(mu)), mu, color=fill_color, ls='solid', lw=0.6)
    ax.fill_between(range(len(mu)),
                    mu + sigma,
                    mu - sigma,
                    alpha=0.3,
                    edgecolor=fill_color,
                    facecolor=fill_color)
    if title:
        ax.set_title(title, fontsize=8.5, pad=2.5)
    ax.grid(True, alpha=0.3, lw=0.3)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xticks(range(0, 120, 20))
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])


def plot_exps(env):
    _, ax = plt.subplots()
    data, rewards = read_data(f'logs', env_name=env, window=7)
    plot_ax(ax, data, colors[0], title=env, label=f'{np.mean(rewards):.2f}±{np.std(rewards):.2f}')
    ax.legend(fontsize=7, loc='lower right')
    plt.savefig(f'imgs/{env}.png')


if __name__ == '__main__':
    env = 'hopper-medium-v2'
    plot_exps(env)
