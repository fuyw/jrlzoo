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


def smooth(x, window=3):
    y = np.ones(window)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x.reshape(-1, 1)


# read experiment res with different seeds
def read_data(env_name='Ant-v2', col='reward', window=7, num=5):
    res_lst = []
    for i in range(num):
        df = pd.read_csv(f'logs/{env_name}/{i}.csv', index_col=0).set_index('step')
        x = df[col].values
        res_lst.append(smooth(x, window=window))
    res_lst = np.concatenate(res_lst, axis=-1)
    return res_lst


def read_np_data(env_name='Ant-v2', col='reward', window=7):
    res_lst = []
    for i in range(3):
        x = np.load(f'logs/results/TD3_{env_name}_{i}.npy')
        res_lst.append(smooth(x, window=window))
    res_lst = np.concatenate(res_lst, axis=-1)
    return res_lst


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
    ax.set_xticks(range(0, 240, 40))
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])


def plot_exp(seeds=True):
    # matplotlib plot setting
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad']='0.1'
    mpl.rcParams['ytick.major.pad']='0'

    envs = ['HalfCheetah', 'Walker2d', 'Hopper']
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    for idx, env in enumerate(envs):
        ax = axes[idx]
        data_jax = read_data(env_name=f'{env}-v2', num=10)
        if seeds:
            for i in range(data_jax.shape[-1]):
                ax.plot(range(data_jax.shape[0]), data_jax[:, i], label=f'{i}')
            ax.legend()
        else:
            plot_ax(ax, data_jax, colors[0], title=f'{env}')
        ax.set_xlabel('Steps (1e6)')
        ax.set_title(f'{env}')
    plt.tight_layout()
    plt.savefig('imgs/ten_seeds_td3.png', dpi=720)



if __name__ == '__main__':
    plot_exp()
