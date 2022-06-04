import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('imgs', exist_ok=True)
mujoco_tasks = ["hopper-v2", "walker2d-v2", "halfcheetah-v2"]


def smooth(x, window=3):
    y = np.ones(window)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x.reshape(-1, 1)


def read_data(log_dir, env_name='hopper-v2', col='reward', window=1):
    res_lst = []
    csv_files = [i for i in os.listdir(f'{log_dir}/{env_name}') if '.csv' in i]
    for csv_file in csv_files:
        df = pd.read_csv(f'{log_dir}/{env_name}/{csv_file}',
                         index_col=0).set_index('step')
        if 'v2' in env_name:
            plot_idx = [20000 * i for i in range(101)]  # plot every 1e4 steps
            x = df.loc[plot_idx, col].values
        else:
            x = df[col].values
        res_lst.append(smooth(x, window=window))
    res_lst = np.concatenate(res_lst, axis=-1)
    return res_lst


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


def plot_exps(tasks, fname="mujoco", nrows=3, ncols=3, smooth_window=1):
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad'] = '0.1'
    mpl.rcParams['ytick.major.pad'] = '0'

    _, axes = plt.subplots(nrows=nrows,
                           ncols=ncols,
                           figsize=(ncols * 4, nrows * 4))
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    for idx, env_name in enumerate(tasks):
        ax = axes[idx]
        data = read_data(log_dir='logs',
                         env_name=env_name,
                         window=smooth_window)
        plot_ax(ax, data, 'b', title=env_name)
    plt.tight_layout()
    plt.savefig(f'imgs/{fname}.png', dpi=360)


if __name__ == '__main__':
    plot_exps(mujoco_tasks, "mujoco", nrows=1, ncols=3, smooth_window=7)
    # plot_exps(antmaze_tasks, "antmaze", 2, 3)
