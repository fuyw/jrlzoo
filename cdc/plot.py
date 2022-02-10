import gym
import d4rl
import pandas as pd
import matplotlib.pyplot as plt


_, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
for i, env_name in enumerate(['halfcheetah-medium', 'halfcheetah-medium-replay', 'halfcheetah-medium-expert',
                              'hopper-medium', 'hopper-medium-replay', 'hopper-medium-expert',
                              'walker2d-medium', 'walker2d-medium-replay', 'walker2d-medium-expert']):
    ax = axes[i//3][i%3]
    res_idx = range(955000, 1005000, 5000)
    plot_idx = [10000 * i for i in range(101)]

    env = gym.make(f'{env_name}-v2')

    df = pd.read_csv(f'../benchmarks/cdc-batch-rl/log_dir/{env_name}-v2_cdc_dummy/eval.csv', dtype={'total_timesteps':int}, index_col=0)
    df['score'] = df['eprewmean'].apply(lambda x: env.get_normalized_score(x)) * 100
    x = df.loc[plot_idx, 'score'].values
    rew = df.loc[res_idx, 'score'].mean()
    ax.plot(range(len(x)), x, label=f'torch_{rew:.2f}')

    try:
        df = pd.read_csv(f'logs/{env_name}-v2/s0.csv', index_col=0).set_index('step')
        x = df.loc[plot_idx, 'reward'].values 
        rew = df.loc[res_idx, 'reward'].mean()
        ax.plot(range(len(x)), x, label=f'cdc0_{rew:.2f}')
    except Exception as e:
        pass

    try:
        df = pd.read_csv(f'logs/{env_name}-v2/s0_sample.csv', index_col=0).set_index('step')
        x = df.loc[plot_idx, 'reward'].values
        rew = df.loc[res_idx, 'reward'].mean()
        ax.plot(range(len(x)), x, label=f'cdc1_{rew:.2f}')
    except Exception as e:
        pass

    ax.legend()
    ax.set_title(f'{env_name}-v2')
plt.savefig('cdc.png')    
