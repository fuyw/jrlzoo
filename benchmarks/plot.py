import numpy as np
import matplotlib.pyplot as plt

env = "halfcheetah-random"
x_td3bc = np.load(f"TD3_BC/results/TD3_BC_{env}-v0_0.npy")
x_td3 = np.load(f"TD3_BC/results/TD3_{env}-v0_0.npy")

plt.plot(range(len(x_td3)), x_td3, label="td3")
plt.plot(range(len(x_td3bc)), x_td3bc, label="td3bc")
plt.legend()
plt.savefig('compare.png')
