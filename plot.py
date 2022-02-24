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


DIRS = {
    'cql': '/usr/local/data/yuweifu/jaxrl/cql/logs_',
    'iql': '/usr/local/data/yuweifu/jaxrl/iql/logs',
    'td3bc': '/usr/local/data/yuweifu/jaxrl/td3bc/logs',
}


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
        df = pd.read_csv(f'{log_dir}/{env_name}/{csv_file}',
                         index_col=0).set_index('step')
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


def plot_exps():
    # matplotlib plot setting
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad'] = '0.1'
    mpl.rcParams['ytick.major.pad'] = '0'

    _, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    for algo_idx, algo in enumerate(DIRS):
        for idx, env in enumerate(tasks):
            ax = axes[idx // 3][idx % 3]
            data, rewards = read_data(DIRS[algo], env_name=env, window=7)
            plot_ax(ax, data, colors[algo_idx], title=env,
                    label=f'{algo} ({np.mean(rewards):.2f}±{np.std(rewards):.2f})')
            ax.legend(fontsize=7, loc='lower right')

    # add combo result
    for i, j, env in [
            (0, 0, 'halfcheetah-medium-v2'),
            (0, 1, 'halfcheetah-medium-replay-v2'),
            (0, 2, 'halfcheetah-medium-expert-v2'),
            (1, 0, 'hopper-medium-v2'),
            (1, 1, 'hopper-medium-replay-v2'),
            (1, 2, 'hopper-medium-expert-v2'),
            (2, 0, 'walker2d-medium-v2'),
            (2, 1, 'walker2d-medium-replay-v2'),
            (2, 2, 'walker2d-medium-expert-v2'),
        ]:
        ax = axes[i][j]
        data, rewards = read_data('combo/logs', env, window=7)
        plot_ax(ax, data, colors[4], title=env, label=f'combo ({np.mean(rewards):.2f}±{np.std(rewards):.2f})')
        ax.legend(fontsize=7, loc='lower right')


    plt.savefig('compare_algo.png', dpi=540)


if __name__ == '__main__':
    plot_exps()
