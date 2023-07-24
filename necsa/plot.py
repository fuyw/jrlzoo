import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns
colors = sns.color_palette('tab10')
os.makedirs('imgs', exist_ok=True)
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']


def smooth(x, window=3):
    y = np.ones(window)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x.reshape(-1, 1)


# read experiment res with different seeds
def read_data(logdir, col='reward', window=7):
    res_lst, rewards = [], []
    csv_files = [i for i in os.listdir(logdir) if '.csv' in i]
    for csv_file in csv_files:
        df = pd.read_csv(f'{logdir}/{csv_file}', index_col=0).set_index('step')
        assert len(df) == 106
        max_step = df.index[-1]
        gap_step = max_step // 100
        plot_idx = np.arange(0, max_step+gap_step, gap_step)
        plot_df = df.loc[plot_idx]
        x = plot_df[col].values
        res_lst.append(smooth(x, window=window))
        rewards.append(df.iloc[-10:]["reward"].mean())
    res_lst = np.concatenate(res_lst, axis=-1)
    return res_lst, rewards


def plot_ax(ax, data, fill_color, title=None, label=None):
    multiple = 10_000
    sigma = np.std(data, axis=1)
    mu = np.mean(data, axis=1)
    if label:
        ax.plot(np.arange(len(mu))*multiple,
                mu,
                color=fill_color,
                ls='solid',
                lw=2,
                label=label)
    else:
        ax.plot(np.arange(len(mu))*multiple, mu, color=fill_color, ls='solid', lw=0.6)
    ax.fill_between(np.arange(len(mu))*multiple,
                    mu + sigma,
                    mu - sigma,
                    alpha=0.1,
                    edgecolor=fill_color,
                    facecolor=fill_color)
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    ax.grid(True, alpha=0.3, lw=0.3)


def plot_exp(envs, exp_name, fdir="logs"):
    # matplotlib plot setting
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad'] = '0.1'
    mpl.rcParams['ytick.major.pad'] = '0'

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    for idx, env in enumerate(envs):
        ax = axes[idx]
        data, rewards = read_data(logdir=f'{fdir}/{env.lower()}', window=10)
        plot_ax(ax, data, colors[0], title=f'{env}', label=f"{np.mean(rewards):.1f}±{np.std(rewards):.1f}")
    plt.tight_layout()
    plt.savefig(f'imgs/{exp_name}.png', dpi=560)


def plot_algos(envs):
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad'] = '0.1'
    mpl.rcParams['ytick.major.pad'] = '0'

    ALGOS = ["TD3"]
    results = {}
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    plt.subplots_adjust(hspace=0.2, wspace=0.5)
    fdirs = {
        "TD3": "curves/td3",
    }
    for idx, env in enumerate(envs):
        ax = axes[idx]
        for a_idx, algo in enumerate(ALGOS):
            data, rewards = read_data(logdir=f'{fdirs[algo]}/{env}', window=10)
            results[(env, algo)] = f"{np.mean(rewards):.1f}±{np.std(rewards):.1f}"
            plot_ax(ax, data, colors[a_idx], title=f'{env}', label=algo)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f'imgs/baselines.png', dpi=480)

    for algo in ["TD3"]:
        print(f"{algo}:")
        for env in envs:
            print(f"{env}: {results[(env, algo)]}")
        print("\n")


if __name__ == '__main__':
    os.makedirs('imgs', exist_ok=True)
    envs = ["Walker2d-v3", "Hopper-v3", "Humanoid-v3"]
    plot_algos(envs)
