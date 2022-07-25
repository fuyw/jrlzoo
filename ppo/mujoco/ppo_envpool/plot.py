import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("imgs", exist_ok=True)

env_names = ["Pong", "Breakout"]

_, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
for i, env_name in enumerate(env_names):
    ax = axes[i]
    fnames = [
        i for i in os.listdir(f"logs/{env_name}NoFrameskip-v4") if ".csv" in i
    ]
    for fname in fnames:
        actor_num = fname.split("_")[2][-1]
        df = pd.read_csv(f"logs/{env_name}NoFrameskip-v4/{fname}")
        ax.plot(df["frame"].values * 1000,
                df["reward"].values,
                label=f"{actor_num} actors")
    ax.legend()
    ax.set_title(env_name)
plt.savefig(f"imgs/ppo.png", dpi=360)
