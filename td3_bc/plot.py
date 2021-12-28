import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



envs=[
    "halfcheetah-random-v0",
    "hopper-random-v0",
    "walker2d-random-v0",
    "halfcheetah-medium-v0",
    "hopper-medium-v0",
    "walker2d-medium-v0",
    "halfcheetah-expert-v0",
    "hopper-expert-v0",
    "walker2d-expert-v0",
    "halfcheetah-medium-expert-v0",
    "hopper-medium-expert-v0",
    "walker2d-medium-expert-v0",
    "halfcheetah-medium-replay-v0",
    "hopper-medium-replay-v0",
    "walker2d-medium-replay-v0"
]

_, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 12))
for idx, env in enumerate(envs):
    ax = axes[idx // 5][idx % 5]
    seed = 0
    df = pd.read_csv(f'/usr/local/data/yuweifu/jaxrl/td3_bc/td3_logs/td3bc_{env}/{seed}.csv', index_col=0)
    ax.plot(range(200), x_np, label='torch')
    ax.legend(fontsize=4)
    ax.set_title(env, fontsize=5)


plt.savefig('compare.png')
