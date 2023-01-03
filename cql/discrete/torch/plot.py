import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import CQLAgent
from utils import eval_policy, register_custom_envs


def plot_params():
    alphas = [11, 1.0, 3.0, 5.0, 10.0]
    df_dict = dict()
    for alpha in alphas:
        df = pd.read_csv(f"logs/cql/cql_alpha{alpha}.csv", index_col=0)
        df_dict[alpha] = df

    cols   = ["reward", "mse_loss", "cql_loss", "avg_ood_Q", "avg_Q", "avg_target_Q"]
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
    for alpha in alphas:
        for idx, col in enumerate(cols):
            ax = axes[idx//3][idx%3]
            df = df_dict[alpha]
            ax.plot(df["step"].values/1e5, df[col].values, label=f"{alpha}")
            ax.legend()
            ax.set_title(col)
    plt.savefig("demo.png", dpi=360)


def save_trajs():
    register_custom_envs()
    env = gym.make("PointmassHard-v2")
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.n
    agent = CQLAgent(obs_dim=obs_dim, act_dim=act_dim)
    for i in range(1, 6):
        agent.load(f"saved_models/online_cql/cql_a0.0_{i}")
        reward = eval_policy(agent, env, eval_episodes=4)
        print(f"ckpt{i} reward = {reward:.2f}")
    dqn_trajs = env.save_trajectories()

    for i in range(1, 6):
        agent.load(f"saved_models/online_cql/cql_a3.0_{i}")
        reward = eval_policy(agent, env, eval_episodes=4)
        print(f"ckpt{i} reward = {reward:.2f}")
    cql_trajs = env.save_trajectories()

    env.plot_trajectories(dqn_trajs, cql_trajs, "imgs/cql_trajs")


def plot_heatmap():
    walls = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    dataset = np.load("buffers/pointmass.npz")
    L, X = dataset["ptr"], 11
    observations = dataset["observations"][:L] * 22

    cnts = np.zeros_like(walls)
    for obs in observations:
        (i, j) = np.floor(obs).astype(int)
        if j == 21:
            cnts[i][21] += 1.0
        else:
            cnts[i][21 - j] += 1.0
    cnts = cnts.clip(0, 150)
    fig, ax = plt.subplots()
    im = ax.imshow(cnts.T, cmap="YlGn")
    plt.fill_between([0-X, 4-X], [11-X, 11-X], [13-X, 13-X], color="gray")
    plt.fill_between([6-X, 18-X], [11-X, 11-X], [13-X, 13-X], color="gray")
    plt.fill_between([20-X, 22-X], [11-X, 11-X], [13-X, 13-X], color="gray")
    plt.fill_between([10-X, 12-X], [13-X, 13-X], [18-X, 18-X], color="gray")
    plt.fill_between([10-X, 12-X], [20-X, 20-X], [22-X, 22-X], color="gray")
    plt.fill_between([12-X, 14-X], [6-X, 6-X], [10-X, 10-X], color="gray")
    plt.fill_between([12-X, 14-X], [0-X, 0-X], [4-X, 4-X], color="gray")
    plt.xticks([])
    plt.yticks([])
    cbarlabel = ""
    cbar_kw = {}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=360)
    plt.close()


def res():
    Data = [
        [[10, 11], [0., 0.], [1, 1]],
        [[11, 12], [0, 0.], [1, 1]],
        [[10, 11], [1, 1], [2, 2]],
        [[11, 12], [1, 1], [2, 2]],
        [[10, 11], [4, 4], [5, 5]],
        [[11, 12], [4, 4], [5, 5]],
        [[10, 11], [5, 5], [6, 6]],
        [[11, 12], [5, 5], [6, 6]],
        [[10, 11], [6, 6], [7, 7]],
        [[11, 12], [6, 6], [7, 7]],
        [[10, 11], [7, 7], [8, 8]],
        [[11, 12], [7, 7], [8, 8]],
        [[10, 11], [8, 8], [9, 9]],
        [[11, 12], [8, 8], [9, 9]],
        [[10, 11], [9, 9], [10, 10]],
        [[11, 12], [9, 9], [10, 10]],
        [[0, 1], [10, 10], [11, 11]],
        [[1, 2], [10, 10], [11, 11]],
        [[2, 3], [10, 10], [11, 11]],
        [[3, 4], [10, 10], [11, 11]],
        [[6, 7], [10, 10], [11, 11]],
        [[7, 8], [10, 10], [11, 11]],
        [[8, 9], [10, 10], [11, 11]],
        [[9, 10], [10, 10], [11, 11]],
        [[10, 11], [10, 10], [11, 11]],
        [[11, 12], [10, 10], [11, 11]],
        [[12, 13], [10, 10], [11, 11]],
        [[13, 14], [10, 10], [11, 11]],
        [[14, 15], [10, 10], [11, 11]],
        [[15, 16], [10, 10], [11, 11]],
        [[16, 17], [10, 10], [11, 11]],
        [[17, 18], [10, 10], [11, 11]],
        [[20, 21], [10, 10], [11, 11]],
        [[21, 22], [10, 10], [11, 11]],
        [[0., 1], [11, 11],[12, 12]],
        [[1, 2], [11, 11], [12, 12]],
        [[2, 3], [11, 11], [12, 12]],
        [[3, 4], [11, 11], [12, 12]],
        [[6, 7], [11, 11], [12, 12]],
        [[7, 8], [11, 11], [12, 12]],
        [[8, 9], [11, 11], [12, 12]],
        [[9, 10], [11, 11], [12, 12]],
        [[10, 11], [11, 11],[12, 12]],
        [[11, 12], [11, 11],[12, 12]],
        [[12, 13], [11, 11], [12, 12]],
        [[13, 14], [11, 11], [12, 12]],
        [[14, 15], [11, 11], [12, 12]],
        [[15, 16], [11, 11], [12, 12]],
        [[16, 17], [11, 11], [12, 12]],
        [[17, 18], [11, 11], [12, 12]],
        [[20, 21], [11, 11], [12, 12]],
        [[21, 22], [11, 11], [12, 12]],
        [[12, 13], [12, 12], [13, 13]],
        [[13, 14], [12, 12], [13, 13]],
        [[12, 13], [13, 13], [14, 14]],
        [[13, 14], [13, 13], [14, 14]],
        [[12, 13], [14, 14], [15, 15]],
        [[13, 14], [14, 14], [15, 15]],
        [[12, 13], [15, 15], [16, 16]],
        [[13, 14], [15, 15], [16, 16]],
        [[12, 13], [18, 18], [19, 19]],
        [[13, 14], [18, 18], [19, 19]],
        [[12, 13], [19, 19], [20, 20]],
        [[13, 14], [19, 19], [20, 20]],
        [[12, 13], [20, 20], [21, 21]],
        [[13, 14], [20, 20], [21, 21]],
        [[12, 13], [21, 21], [22, 22]],
        [[13, 14], [21, 21], [22, 22]],
    ]
    dataset = np.load("buffers/pointmass.npz")
    L, X = dataset["ptr"], 11
    observations = dataset["observations"][:L] * 22
    cnts = np.zeros(shape=(22, 22))
    for obs in observations:
        (i, j) = np.floor(obs).astype(int)
        if j == 21:
            cnts[i][21] += 1.0
        else:
            cnts[i][21 - j] += 1.0
    cnts = cnts.clip(0, 150)
    fig, ax = plt.subplots()
    im = ax.imshow(cnts.T, cmap="YlGn")

    L = len(Data)
    for i in range(L):
        x, y0, y1 = Data[i]
        x = np.array(x) - 0.5
        y0 = 22 - np.array(y0) - 0.5
        y1 = 22 - np.array(y1) - 0.5
        plt.fill_between(x, y0, y1, color="gray")
    plt.xticks([])
    plt.yticks([])
    cbarlabel = ""
    cbar_kw = {}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=360)
     
     
if __name__:
    # save_trajs()
    # plot_heatmap()
    res()
