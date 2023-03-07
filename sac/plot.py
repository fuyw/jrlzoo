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
                lw=0.6,
                label=label)
        ax.legend()
    else:
        ax.plot(np.arange(len(mu))*multiple, mu, color=fill_color, ls='solid', lw=0.6)
    ax.fill_between(np.arange(len(mu))*multiple,
                    mu + sigma,
                    mu - sigma,
                    alpha=0.3,
                    edgecolor=fill_color,
                    facecolor=fill_color)
    if title:
        ax.set_title(title, fontsize=8.5, pad=4)
    ax.grid(True, alpha=0.3, lw=0.3)


def plot_exp(envs, exp_name):
    # matplotlib plot setting
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad'] = '0.1'
    mpl.rcParams['ytick.major.pad'] = '0'

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    for idx, env in enumerate(envs):
        ax = axes[idx]
        data, rewards = read_data(logdir=f'logs/{env.lower()}', window=10)
        plot_ax(ax, data, colors[0], title=f'{env}', label=f"{np.mean(rewards):.1f}±{np.std(rewards):.1f}")
    plt.tight_layout()
    plt.savefig(f'imgs/{exp_name}.png', dpi=560)


def compare_tandem(env_name, cols):
    fnames = [i for i in os.listdir(f"logs/{env_name}") if ".csv" in i]
    dfs = [pd.read_csv(f"logs/{env_name}/{fname}", index_col=0).set_index("step") for fname in fnames]
    _, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(4.5*len(cols), 4))
    plt_idx = np.arange(0, 1010000, 10000)
    for i, col in enumerate(cols):
        ax = axes[i]
        data1 = pd.concat([df.loc[plt_idx, f"{col}_1"] for df in dfs], axis=1)
        data2 = pd.concat([df.loc[plt_idx, f"{col}_2"] for df in dfs], axis=1)
        mu1, std1 = data1.mean(1), data1.std(1)
        mu2, std2 = data2.mean(1), data2.std(1)
        ax.plot(plt_idx, mu1, label="online agent")
        ax.fill_between(plt_idx, mu1+std1, mu1-std1, alpha=0.1)
        ax.plot(plt_idx, mu2, label="tandem agent")
        ax.fill_between(plt_idx, mu2+std2, mu2-std2, alpha=0.1)
        ax.set_title(col)
        if (col == cols[-1]): ax.legend()
    plt.tight_layout()
    plt.savefig(f"imgs/tandem_{env_name}.png")


if __name__ == '__main__':
    os.makedirs('imgs', exist_ok=True)
    dmc_envs = ["cheetah-run", "quadruped-run", "humanoid-run", "hopper-hop"]
    mj_envs = ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2', 'Ant-v2']
    # plot_exp(dmc_envs, 'dmc')
    # plot_exp(mj_envs, 'mujoco')
    compare_tandem("cheetah-run", cols=["reward", "q1", "critic_loss", "alpha_loss"])
