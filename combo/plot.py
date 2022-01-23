import pandas as pd
import matplotlib.pyplot as plt

env_name = 'hopper-medium-v2'

cols = ['evaluation/return-average', 'Q_loss']
res_df = pd.read_csv(f'logs/{env_name}/progress.csv')
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
for i in range(2):
    col = cols[i]
    ax = axes[i]
    ax.plot(res_df['train-steps'], res_df[col])
    if col == 'Q_loss':
        ax.set_yscale('log')
    ax.set_title(col)
plt.savefig('baseline_hopper.png')
plt.close()

