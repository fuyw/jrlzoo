import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('imgs', exist_ok=True)
colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c",
          "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

env_names = ['halfcheetah', 'hopper', 'walker2d']
levels = ['medium', 'medium-replay', 'medium-expert']
tasks = [f'{env_name}-{level}-v2' for env_name in env_names for level in levels]


def smooth(x, window=3):
    y = np.ones(window)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x.reshape(-1, 1)


def read_data(env_name='hopper-medium-v2', col='reward', window=1):
    rewards = []
    res_lst = []
    for i in range(5):
        df = pd.read_csv(f'logs/{env_name}/{i}.csv', index_col=0).set_index('step')
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
        ax.plot(range(len(mu)), mu, color=fill_color, ls='solid', lw=0.6, label=label)
    else:
        ax.plot(range(len(mu)), mu, color=fill_color, ls='solid', lw=0.6)
    ax.fill_between(range(len(mu)), mu+sigma, mu-sigma, alpha=0.3, edgecolor=fill_color, facecolor=fill_color)
    if title:
        ax.set_title(title, fontsize=8.5, pad=2.5)
    ax.grid(True, alpha=0.3, lw=0.3)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xticks(range(0, 120, 20))
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])


def plot_exp():
    # matplotlib plot setting
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad']='0.1'
    mpl.rcParams['ytick.major.pad']='0'

    _, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    for idx, env in enumerate(tasks):
        ax = axes[idx // 3][idx % 3]
        data, rewards = read_data(env_name=f'{env}')
        plot_ax(ax, data, colors[0], title=f'{env}', label=f'{np.mean(rewards):.2f} ({np.std(rewards):.2f})')
        ax.legend(fontsize=7, loc='lower right')

    # patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i, metric in enumerate(metrics)]
    # plt.figlegend(handles=patches, loc='upper center', fontsize=8.5, ncol=len(metrics), frameon=False, bbox_to_anchor=(0.5, 0.95), handlelength=0.7)
    plt.tight_layout()
    plt.savefig('td3bc.png', dpi=720)


if __name__ == '__main__':
    plot_exp()