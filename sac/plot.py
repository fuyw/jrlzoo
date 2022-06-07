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


def smooth(x, window=3):
    y = np.ones(window)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x.reshape(-1, 1)


# read experiment res with different seeds
def read_data(logdir, col='reward', window=7):
    res_lst = []
    csv_files = [
        i for i in os.listdir(logdir) if '.csv' in i
    ]
    for csv_file in csv_files:
        df = pd.read_csv(f'{logdir}/{csv_file}', index_col=0).set_index('step')
        x = df[col].values
        res_lst.append(smooth(x, window=window))
    res_lst = np.concatenate(res_lst, axis=-1)
    return res_lst


def plot_ax(ax, data, fill_color, title=None, label=None):
    multiple = 30_000
    sigma = np.std(data, axis=1)
    mu = np.mean(data, axis=1)
    if label:
        ax.plot(np.arange(len(mu))*multiple,
                mu,
                color=fill_color,
                ls='solid',
                lw=0.6,
                label=label)
    else:
        ax.plot(np.arange(len(mu))*multiple, mu, color=fill_color, ls='solid', lw=0.6)
    ax.fill_between(np.arange(len(mu))*multiple,
                    mu + sigma,
                    mu - sigma,
                    alpha=0.3,
                    edgecolor=fill_color,
                    facecolor=fill_color)
    if title:
        ax.set_title(title, fontsize=8.5, pad=2.5)
    ax.grid(True, alpha=0.3, lw=0.3)
    # ax.xaxis.set_ticks_position('none')
    # ax.yaxis.set_ticks_position('none')
    # ax.set_xticks(range(0, 240, 40))
    # ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])


def plot_exp():
    # matplotlib plot setting
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad'] = '0.1'
    mpl.rcParams['ytick.major.pad'] = '0'

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    for idx, env in enumerate(['HalfCheetah', 'Hopper', 'Walker2d', 'Ant']):
        ax = axes[idx]
        data = read_data(logdir=f'logs/{env.lower()}-v2/ups1', window=1)
        plot_ax(ax, data, colors[0], title=f'{env}-v2')
    plt.tight_layout()
    plt.savefig('imgs/sac.png', dpi=560)


def plot_single(env_name="quadruped-run"):
    cols = ["reward", "time"]
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    for idx, col in enumerate(cols):
        ax = axes[idx]
        data = read_data(env_name, col)
        mu, _ = data.mean(1), data.std(1)
        ax.plot(range(len(mu)), mu)
        ax.set_title(col)
    plt.savefig(f"imgs/{env_name}.png", dpi=320)


if __name__ == '__main__':
    os.makedirs('imgs', exist_ok=True)
    plot_exp()
    # plot_single("quadruped-run")
