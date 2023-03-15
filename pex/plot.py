import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mujoco_tasks = [
    "halfcheetah-random-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "hopper-random-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "walker2d-random-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
]

antmaze_tasks = [
    "antmaze-umaze-v0",
    "antmaze-umaze-diverse-v0",
    "antmaze-medium-play-v0",
    "antmaze-medium-diverse-v0",
    "antmaze-large-play-v0",
    "antmaze-large-diverse-v0",
]

colors = [
    "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#8c564b",
    "#e377c2", "#bcbd22", "#17becf"
]
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

Temps = [3.0, 5.0, 7.0, 10.0]
Taus = [0.7, 0.8, 0.9]


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
        if 'v2' in env_name:
            plot_idx = np.arange(0, 252500, 2500)
            res_idx = np.arange(227500, 252500, 2500)  # eval every 5e3 steps
            x = df.loc[plot_idx, col].values
            rewards.append(df.loc[res_idx, 'reward'].mean())
        else:
            plot_idx = np.arange(0, 275000, 25000)
            x = df.loc[plot_idx, col].values
            rewards.append(df.iloc[-1]['reward'].mean())
        res_lst.append(smooth(x, window=window))
    res_lst = np.concatenate(res_lst, axis=-1)
    rewards = np.array(rewards)
    return res_lst, rewards


def plot_ax(ax,
            data,
            fill_color,
            plt_idx=np.arange(0, 252500, 2500) / 1e5,
            title=None,
            log=False,
            label=None):
    sigma = np.std(data, axis=1)
    mu = np.mean(data, axis=1)
    if label:
        ax.plot(
            plt_idx,
            # np.arange(0, 252500, 2500)/1e5,
            mu,
            color=fill_color,
            ls='solid',
            lw=0.6,
            label=label)
    else:
        ax.plot(
            plt_idx,
            # np.arange(0, 252500, 2500)/1e5,
            mu,
            color=fill_color,
            ls='solid',
            lw=0.6)
    ax.fill_between(
        plt_idx,
        # np.arange(0, 252500, 2500)/1e5,
        mu + sigma,
        mu - sigma,
        alpha=0.05,
        edgecolor=fill_color,
        facecolor=fill_color)
    if title:
        ax.set_title(title, fontsize=8.5, pad=2.5)
    ax.grid(True, alpha=0.3, lw=0.3)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')


#################
# Exp Baselines #
#################
def plot_exps(tasks,
              fname="mujoco",
              nrows=3,
              ncols=3,
              window=1):
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad'] = '0.1'
    mpl.rcParams['ytick.major.pad'] = '0'

    _, axes = plt.subplots(nrows=nrows,
                           ncols=ncols,
                           figsize=(ncols * 4, nrows * 4))
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    total_reward = 0
    for idx, env_name in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        data, rewards = read_data(log_dir=f'logs',
                                  env_name=env_name,
                                  window=window)
        if "v2" in env_name:
            plt_idx = np.arange(0, 252500, 2500) / 1e5
        else:
            plt_idx = np.arange(0, 275000, 25000) / 1e5
        plot_ax(ax,
                data,
                'b',
                plt_idx,
                title=env_name,
                label=f'({np.mean(rewards):.1f}±{np.std(rewards):.1f})')
        ax.legend(fontsize=7, loc='lower right')
        exp_reward = float(f'{np.mean(rewards):.1f}')
        total_reward += exp_reward
    plt.tight_layout()
    print(f"Total reward: {total_reward:.1f}")
    plt.savefig(f'imgs/{fname}.png', dpi=360)


if __name__ == '__main__':
    os.makedirs("imgs", exist_ok=True)
    plot_exps(mujoco_tasks, fname="mujoco", nrows=3, ncols=3, window=11)
    # plot_exps(antmaze_tasks, prefix="exp_antmaze_gde/gde_lb0.3", fname="antmaze", nrows=2, ncols=3, window=1)
