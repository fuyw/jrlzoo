import pandas as pd
import matplotlib.pyplot as plt

env_name = 'hopper-medium-v2'


for target in ['reward', 'next_obs', 'action']:
    res = []
    for algo in ['combo', 'cql', 'td3bc', 'random']:
        df = pd.read_csv(f'res/{env_name}/{algo}_probe_{target}.csv', index_col=0)
        res.append((algo, df['cv_loss'].mean(), df['cv_loss'].std()))
    res_df = pd.DataFrame(res, columns=['algo', 'mean', 'std'])
    res_df.to_csv(f'res/{target}.csv')
