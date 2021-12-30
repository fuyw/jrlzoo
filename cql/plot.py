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
def read_data(env_name='Ant-v2', col='reward', window=7):
    res_lst = []
    for i in range(3):
        df = pd.read_csv(f'logs/{env_name}/{i}.csv', index_col=0).set_index('step')
        x = df[col].values
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


def plot_exp():
    # matplotlib plot setting
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.pad']='0.1'
    mpl.rcParams['ytick.major.pad']='0'

    # _, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    # plt.subplots_adjust(hspace=0.2, wspace=0.15)
    # for idx, env in enumerate(['HalfCheetah', 'Hopper', 'Walker2d', 'Ant']):
    #     ax = axes[idx // 2][idx % 2]
    #     data_jax = read_data(env_name=f'{env}-v2')
    #     plot_ax(ax, data_jax, colors[0], title=f'{env}', label='jax_td3')
    #     ax.legend(fontsize=7, loc='upper left')

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    for idx, env in enumerate(['HalfCheetah', 'Hopper']):
        ax = axes[idx]
        data_jax = read_data(env_name=f'{env}-v2')
        plot_ax(ax, data_jax, colors[0], title=f'{env}')

    # patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i, metric in enumerate(metrics)]
    # plt.figlegend(handles=patches, loc='upper center', fontsize=8.5, ncol=len(metrics), frameon=False, bbox_to_anchor=(0.5, 0.95), handlelength=0.7)
    plt.tight_layout()
    plt.savefig('sac.png', dpi=720)


def plot_single_run():
    df = pd.read_csv('/usr/local/data/yuweifu/jaxrl/cql/logs/hopper-medium-v2/s1.csv')
    x = df['reward'].values
    r1 = x[-10:].mean()
    plt.plot(range(len(x)), x, label=f'my: {r1:.2f}')

    df = pd.read_csv('/usr/local/data/yuweifu/jaxrl/benchmarks/logs/hopper-medium-v2/hopper-medium-s1.csv')
    df = df[['sac/average_qf1', 'sac/average_qf2', 'sac/policy_loss', 'sac/qf1_loss', 'sac/qf2_loss', 'average_return', 'average_normalizd_return']]
    idx = [i*5-1 for i in range(1, len(df) // 5 +1)]
    df = df.loc[idx] 
    x = (df['average_normalizd_return'] * 100).values
    r2 = x[-10:].mean()
    plt.plot(range(len(x)), x, label=f'jaxcql: {r2:.2f}')
    plt.legend()
    plt.savefig('1.png')


def plot_one_env():
    # res = []
    # for i in range(1, 10):
    #     df = pd.read_csv(f'/usr/local/data/yuweifu/jaxrl/benchmarks/logs/hopper-medium-v2/hopper-medium-s{i}.csv', index_col=0)
    #     df = df[['sac/average_qf1', 'sac/average_qf2', 'sac/policy_loss', 'sac/qf1_loss', 'sac/qf2_loss', 'average_return', 'average_normalizd_return']]
    #     idx = [i*5-1 for i in range(1, len(df) // 5 +1)]
    #     df = df.loc[idx]
    #     df['average_normalizd_return'] *= 100
    #     res.append(df['average_normalizd_return'].values)

    # data = np.array(res).T
    # sigma = np.std(data, axis=1)
    # mu = np.mean(data, axis=1)
    # _, ax = plt.subplots()
    # rew = data[-10:, :].reshape(-1).mean()
    # ax.plot(range(len(mu)), mu, ls='solid', lw=0.6, label=f'jaxrl({rew:.2f})')
    # ax.fill_between(range(len(mu)), mu+sigma, mu-sigma, alpha=0.3)  #, edgecolor=fill_color, facecolor=fill_color)

    res = []
    for i in range(4):
        df = pd.read_csv(f'/usr/local/data/yuweifu/jaxrl/cql/logs/hopper-medium-v2_glurot/Backup_entropy0_s{i}.csv', index_col=0)
        res.append(df['reward'].values)

    data = np.array(res).T
    sigma = np.std(data, axis=1)
    mu = np.mean(data, axis=1)
    _, ax = plt.subplots()
    rew = data[-10:, :].reshape(-1).mean()
    ax.plot(range(len(mu)), mu, ls='solid', lw=0.6, label=f'jaxrl({rew:.2f})')
    ax.fill_between(range(len(mu)), mu+sigma, mu-sigma, alpha=0.3)  #, edgecolor=fill_color, facecolor=fill_color)


    res2 = []
    for i in [0,1,2,3,4,8]:
        df = pd.read_csv(f'/usr/local/data/yuweifu/jaxrl/cql/logs/hopper-medium-v2/s{i}.csv', index_col=0)
        res2.append(df['reward'].values)
    data2 = np.array(res2).T
    sigma2 = np.std(data2, axis=1)
    mu2 = np.mean(data2, axis=1)
    rew2 = data2[-10:, :].reshape(-1).mean()
    ax.plot(range(len(mu2)), mu2, ls='solid', color='red', lw=0.6, label=f'myjax({rew2:.2f})')
    ax.fill_between(range(len(mu2)), mu2+sigma2, mu2-sigma2, alpha=0.3, edgecolor='red', facecolor='red')
    ax.legend()
    plt.savefig('b.png')


def plot_seeds():
    fdir = '/usr/local/data/yuweifu/jaxrl/cql/logs/hopper-medium-expert-v2'
    _, ax = plt.subplots()
    for seed in [0, 1]:
        df = pd.read_csv(f'{fdir}/s{seed}.csv')
        if len(df) == 201:
            reward = df['reward'].iloc[-10:].mean()
            ax.plot(range(len(df)), df['reward'], label=f's{seed}: {reward:.2f}')
    plt.legend()
    plt.savefig('seeds.png')


if __name__ == '__main__':
    # plot_exp()
    # plot_single_run()
    plot_seeds()
