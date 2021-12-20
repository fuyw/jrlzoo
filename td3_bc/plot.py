import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

env = 'halfcheetah-random-v0'
seed = 0
df = pd.read_csv(f'/usr/local/data/yuweifu/jaxrl/td3_bc/td3_logs/td3bc_{env}/{seed}.csv', index_col=0)
x_np = np.load(f'/usr/local/data/yuweifu/jaxrl/td3_bc/TD3_BC/results/TD3_BC_{env}_{seed}.npy')

plt.plot(range(200), x_np, label='torch')
plt.plot(range(200), df['reward'].values[1:], label='jax')
plt.legend()
plt.savefig('compare.png')
