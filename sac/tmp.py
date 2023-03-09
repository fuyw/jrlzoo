import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fdir = "/home/yuwei/jrlzoo/sac/backup/curves/sac/max_steps_2000000_resets_False_config.updates_per_step_1"

for env_name in ["hopper-hop", "cheetah-run", "humanoid-run", "quadruped-run"]:
    res, rewards = [], []
    for i in range(10):
        data = [[0, 0]]
        with open(f"{fdir}/{env_name}/{i}.txt", "r") as f:
            for i in f:
                data.append(i.strip().split(" "))
        df = pd.DataFrame(data, columns=["step", "reward"])
        df["step"] = df["step"].astype(int)
        df["reward"] = df["reward"].astype(float)
        df = df.set_index("step")
        plt_idx = np.arange(0, 1010000, 10000)
        reward_idx = np.arange(955000, 1005000, 5000)
        rewards.append(df.loc[reward_idx, "reward"].mean())
        res.append(df.loc[plt_idx, "reward"])
    res_df = pd.concat(res, axis=1)

    _, ax = plt.subplots()
    mu = res_df.mean(1).values
    std = res_df.std(1).values
    r_mu = np.mean(rewards)
    r_std = np.std(rewards)
    ax.plot(res_df.index, mu)
    ax.fill_between(res_df.index, mu+std, mu-std, alpha=0.5)
    ax.set_title(f"{env_name}: {r_mu:.2f}({r_std:.2f})")
    plt.savefig(f"{env_name}.png", dpi=480)
