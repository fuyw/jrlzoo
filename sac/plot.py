import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


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
    multiple = 30_000 if 'v2' in title else 10_000
    sigma = np.std(data, axis=1)
    mu = np.mean(data, axis=1)
    if label:
        ax.plot(np.arange(len(mu))*multiple,
                mu,
                color=fill_color,
                ls='solid',
                lw=0.6,
                label=label)
        ax.legend(loc='best')
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


def plot_exp(envs, exp_name):
    # matplotlib plot setting
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad'] = '0.1'
    mpl.rcParams['ytick.major.pad'] = '0'

    _, axes = plt.subplots(nrows=1, ncols=len(envs), figsize=(16, 4))
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    for idx, env in enumerate(envs):
        ax = axes[idx]
        data = read_data(logdir=f'logs/{env.lower()}/ups1', window=1)
        rewards = data[-10:].mean(1)
        plot_ax(ax, data, colors[0], title=f'{env}', label=f"{np.mean(rewards):.1f}±{np.std(rewards):.1f}")
    plt.tight_layout()
    plt.savefig(f'imgs/{exp_name}.png', dpi=560)


if __name__ == '__main__':
    os.makedirs('imgs', exist_ok=True)
    # envs = ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2', 'Ant-v2']
    envs = ["cheetah-run", "quadruped-run", "humanoid-run", "hopper-hop"]
    plot_exp(envs, 'dmc')
