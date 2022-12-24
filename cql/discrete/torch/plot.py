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


def plot_trajs():
    register_custom_envs()
    env = gym.make("PointmassHard-v2")
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.n
    agent = CQLAgent(obs_dim=obs_dim, act_dim=act_dim)
    raw_reward = eval_policy(agent, env)
    agent.load("saved_models/cql/cql_s42")
    for _ in range(21):
        _ = eval_policy(agent, env)
    env.plot_trajectories("imgs/cql_trajs.png")


if __name__:
    plot_trajs()
