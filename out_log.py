import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

steps = []
lines = []
idxes = []
with open('800.out', 'r') as f:
    for idx, line in tqdm(enumerate(f)):
        content = line.strip()
        lines.append(content)
        if content[:24] == 'Diagnostics -- iteration':
            idxes.append(idx)

res = []
for idx in tqdm(idxes):
    tmp_res = []
    for i in range(8):
        i_line = lines[idx + i + 1]
        split_i_line = i_line.split(',')
        for split_i in split_i_line:
            tmp_res.append(split_i.split(':')[1].strip())
    res.append(tmp_res)



res_df = pd.DataFrame(res, columns=[
    'real_batch_obs', 'model_batch_obs', 'real_batch_act', 'model_batch_act',
    'real_batch_rew', 'model_batch_rew', 'real_batch_done', 'model_batch_done',
    'ret', 'steps', 'Q-avg', 'Q-max', 'Q-min', 'Q_loss1', 'Q_loss2',
    'min_Q_loss1', 'min_Q_loss2'])

df1 = pd.read_csv('/usr/local/data/yuweifu/jaxrl/combo/check_tf/logs/hopper-medium-v0/combo_s1_alpha3.0_check.csv')
df1.rename(columns={'step':'steps', 'reward':'ret', 'Q_loss_1': 'Q_loss1',
                   'q1': 'Q-avg', 'cql1_loss': 'min_Q_loss1'}, inplace=True)
df2 = pd.read_csv('/usr/local/data/yuweifu/jaxrl/combo/check_tf/logs/hopper-medium-v0/combo_s42_alpha3.0_check.csv')
df2.rename(columns={'step':'steps', 'reward':'ret', 'Q_loss_1': 'Q_loss1',
                   'q1': 'Q-avg', 'cql1_loss': 'min_Q_loss1'}, inplace=True)
df3 = pd.read_csv('/usr/local/data/yuweifu/jaxrl/combo/check_tf/logs/hopper-medium-v0/s42.csv')
df3.rename(columns={'step':'steps', 'reward':'ret', 'Q_loss_1': 'Q_loss1',
                   'q1': 'Q-avg', 'cql1_loss': 'min_Q_loss1'}, inplace=True)

res_df = res_df.astype(float)
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
for i, col in enumerate(['ret', 'Q-avg', 'Q_loss1', 'min_Q_loss1']):
    ax = axes[i]
    ax.plot(res_df['steps'].values, res_df[col].values, label='baseline')
    ax.plot(df1['steps'].values, df1[col].values, label='s1')
    ax.plot(df2['steps'].values, df2[col].values, label='s42')
    ax.plot(df3['steps'].values, df3[col].values, label='pool')
    ax.legend()
    ax.set_title(col)
plt.savefig('hopper8.png')
plt.close()
