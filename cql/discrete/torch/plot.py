import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import CQLAgent
from utils import eval_policy, register_custom_envs


def plot_params():
    alphas = [0.5, 1.0, 3.0, 5.0, 10.0]
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
    L, X = dataset["ptr"], 0.5
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
    plt.fill_between([12-X, 14-X], [7-X, 7-X], [12-X, 12-X], color="gray")
    plt.fill_between([12-X, 14-X], [0-X, 0-X], [5-X, 5-X], color="gray")
    plt.xticks([])
    plt.yticks([])
    cbarlabel = ""
    cbar_kw = {}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=360)
    plt.close()


if __name__:
    save_trajs()
