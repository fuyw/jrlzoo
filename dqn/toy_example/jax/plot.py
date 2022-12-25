import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# env_name = "MountainCar-v0"
env_name = "CartPole-v1"

colors = [
    "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#8c564b",
    "#e377c2", "#bcbd22", "#17becf"
]

def smooth(x, window=3):
    y = np.ones(window)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x.reshape(-1, 1)


def get_data(arch, env_name, algo, er, col="reward", window=7, seeds=range(5)):
    res = []
    for i in seeds:
        df = pd.read_csv(f"logs_{arch}/{env_name}/{algo}/{algo}_s{i}_{er}.csv", index_col=0)
        x = df[col].values
        res.append(smooth(x, window=window))
    res = np.concatenate(res, axis=-1)
    mu = res.mean(axis=-1)
    std = res.std(axis=-1)
    return mu, std


def plot_all():
    _, ax = plt.subplots()
    i = 0
    plt_idx = range(0, 1515, 15)
    for algo in ["dqn", "ddqn"]:
        for er in ["er", "per"]:
            mu, std = get_data(env_name, algo, er)
            ax.plot(plt_idx, mu, color=colors[i], ls='solid', lw=0.6, label=f"{algo}_{er}")
            ax.fill_between(plt_idx, mu+std, mu-std, alpha=0.3, edgecolor=colors[i], facecolor=colors[i])
            i += 1
            ax.legend()
    plt.savefig(f'{env_name}.png', dpi=360)


plt_idx = range(0, 1515, 15)
def plot_one(algo="dqn", env_name="CartPole-v1", er="er"):
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    for j, arch in enumerate(["L1_64", "L1_128", "L2_64", "L2_128"]):
        for i, col in enumerate(["reward", "Q", "loss"]):
            ax = axes[i]
            mu, std = get_data(arch, env_name, algo, er, col, window=5)
            ax.plot(plt_idx, mu, color=colors[j], ls='solid', lw=0.5, label=f"{arch}")
            ax.fill_between(plt_idx, mu+std, mu-std, alpha=0.3, edgecolor=colors[j], facecolor=colors[j])
            ax.set_title(col)
            ax.set_xlabel("episodes")
            ax.legend()
    plt.tight_layout()
    plt.savefig(f"imgs/{algo}_{env_name}_{er}.png", dpi=360)

def plot_two(algo="ddqn"):
    env_name = "CartPole-v1"
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
    for j, arch in enumerate(["L1_64", "L1_128", "L2_64", "L2_128"]):
        for i, col in enumerate(["reward", "Q", "loss"]):
            ax = axes[0][i]
            mu, std = get_data(arch, env_name, algo, er, col, window=5)
            ax.plot(plt_idx, mu, color=colors[j], ls='solid', lw=0.5, label=f"{arch}")
            ax.fill_between(plt_idx, mu+std, mu-std, alpha=0.3, edgecolor=colors[j], facecolor=colors[j])
            ax.set_title(f"{col} on {env_name}")
            ax.set_xlabel("episodes")
            ax.legend()

    env_name = "MountainCar-v0"
    for j, arch in enumerate(["L1_64", "L1_128", "L2_64", "L2_128"]):
        for i, col in enumerate(["reward", "Q", "loss"]):
            ax = axes[1][i]
            mu, std = get_data(arch, env_name, algo, er, col, window=5)
            ax.plot(plt_idx, mu, color=colors[j], ls='solid', lw=0.5, label=f"{arch}")
            ax.fill_between(plt_idx, mu+std, mu-std, alpha=0.3, edgecolor=colors[j], facecolor=colors[j])
            ax.set_title(f"{col} on {env_name}")
            ax.set_xlabel("episodes")
            ax.legend()
    plt.tight_layout()
    plt.savefig(f"imgs/{algo}_{er}.png", dpi=360)


def compare_X(col="Q"):
    env_name = "CartPole-v1"
    _, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    for j, arch in enumerate(["L1_64", "L1_128", "L2_64", "L2_128"]):
        ax = axes[0][j]
        for i, algo in enumerate(["dqn", "ddqn"]):
            mu, std = get_data(arch, env_name, algo, er, col, window=5)
            ax.plot(plt_idx, mu, color=colors[i], ls='solid', lw=0.5, label=algo)
            ax.fill_between(plt_idx, mu+std, mu-std, alpha=0.3, edgecolor=colors[i], facecolor=colors[i])
        ax.set_title(f"{col} for {arch} on {env_name}")
        ax.set_xlabel("episodes")
        ax.legend() 

    env_name = "MountainCar-v0"
    for j, arch in enumerate(["L1_64", "L1_128", "L2_64", "L2_128"]):
        ax = axes[1][j]
        for i, algo in enumerate(["dqn", "ddqn"]):
            mu, std = get_data(arch, env_name, algo, er, col, window=5)
            ax.plot(plt_idx, mu, color=colors[i], ls='solid', lw=0.5, label=algo)
            ax.fill_between(plt_idx, mu+std, mu-std, alpha=0.3, edgecolor=colors[i], facecolor=colors[i])
        ax.set_title(f"{col} for {arch} on {env_name}")
        ax.set_xlabel("episodes")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"imgs/compare_{col}.png", dpi=360)


def compare_per(col="reward"):
    _, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    for z, (env_name, algo) in enumerate([("CartPole-v1", "dqn"), ("CartPole-v1", "ddqn"),
                                          ("MountainCar-v0", "dqn"), ("MountainCar-v0", "ddqn")]):
        for j, arch in enumerate(["L1_64", "L1_128", "L2_64", "L2_128"]):
            ax = axes[z][j]
            for i, er in enumerate(["er", "per"]):
                mu, std = get_data(arch, env_name, algo, er, col, window=5)
                ax.plot(plt_idx, mu, color=colors[i], ls='solid', lw=0.5, label=f"{arch}_{er}")
                ax.fill_between(plt_idx, mu+std, mu-std, alpha=0.3, edgecolor=colors[i], facecolor=colors[i])
            ax.set_title(f"{col} for {algo} on {env_name}")
            ax.set_xlabel("episodes")
            ax.legend() 

    plt.tight_layout()
    plt.savefig(f"imgs/compare_per.png", dpi=360)



if __name__ == "__main__":
    # env_name = "MountainCar-v0"
    env_name = "CartPole-v1"
    algo = "ddqn"
    er = "er"
    # plot_one(algo, env_name, er)
    # plot_two(algo)
    compare_per("reward")
