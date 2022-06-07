import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_seeds():
    fdir = 'combo/logs/hopper-medium-expert-v2'
    fnames = [i for i in os.listdir(fdir) if i[-4:]=='.csv']

    _, ax = plt.subplots()
    for i, fname in enumerate(fnames):
        df = pd.read_csv(f'{fdir}/{fname}', index_col=0).set_index(['step'])
        plt_idx = range(0, 1010000, 10000)
        ax.plot(plt_idx, df.loc[plt_idx, 'reward'], label=i)
    plt.savefig('compare_seeds.png', dpi=360)


def plot_repr_similarity():
    """
    Index([
        'reward', 'actor_loss', 'actor_repr_norm', 'bc_loss', 'critic_loss',
        'dot_prod_repr_q1', 'dot_prod_repr_q2', 'max_actor_loss',
        'max_actor_repr_norm', 'max_bc_loss', 'max_critic_loss',
        'max_dot_prod_repr_q1', 'max_dot_prod_repr_q2',
        'max_normalized_dot_prod_repr_q1', 'max_normalized_dot_prod_repr_q2',
        'max_q1', 'max_q1_repr_nonzero', 'max_q1_repr_norm', 'max_q2',
        'max_q2_repr_nonzero', 'max_q2_repr_norm', 'max_target_q',
        'min_actor_loss', 'min_actor_repr_norm', 'min_bc_loss',
        'min_critic_loss', 'min_dot_prod_repr_q1', 'min_dot_prod_repr_q2',
        'min_normalized_dot_prod_repr_q1', 'min_normalized_dot_prod_repr_q2',
        'min_q1', 'min_q1_repr_nonzero', 'min_q1_repr_norm', 'min_q2',
        'min_q2_repr_nonzero', 'min_q2_repr_norm', 'min_target_q',
        'normalized_dot_prod_repr_q1', 'normalized_dot_prod_repr_q2', 'q1',
        'q1_repr_nonzero', 'q1_repr_norm', 'q2', 'q2_repr_nonzero',
        'q2_repr_norm', 'target_q', 'actor_kernel_norm', 'actor_output_norm',
        'q1_kernel_norm', 'q1_output_norm', 'q2_kernel_norm', 'q2_output_norm',
        'eval_time', 'time'],
        dtype='object')
    """
    import gym, d4rl
    env_name = "hopper-medium-v2"
    env = gym.make(env_name)
    plt_idx = range(0, 1010000, 10000)
    td3_df = pd.read_csv(f"JaxTD3/logs/{env_name.split('-')[0]}-v2/td3_s0.csv", index_col=0).set_index('step')
    td3_df['reward'] = td3_df['reward'].apply(lambda x: env.get_normalized_score(x)) * 100.0
    td3bc_df = pd.read_csv(f"JaxTD3BC/logs/{env_name}/td3bc_s0.csv", index_col=0).set_index('step')
    offline_td3_df = pd.read_csv(f"JaxTD3BC/logs/{env_name}/td3_s1.csv", index_col=0).set_index('step')
    _, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 16))
    for idx, col in enumerate(['reward', 'critic_loss', 'q1_repr_nonzero', 'q1', 'q1_repr_norm',
                               'dot_prod_repr_q1', 'normalized_dot_prod_repr_q1',
                               'q1_kernel_norm', 'q1_output_norm']):
        ax = axes[idx//3][idx%3]
        if col in ['critic_loss', 'q1', 'dot_prod_repr_q1']:
            ax.set_yscale('log')
        ax.plot(plt_idx, td3_df.loc[plt_idx, col], label=f'[td3] {col}')
        ax.plot(plt_idx, td3bc_df.loc[plt_idx, col], label=f'[td3bc] {col}')
        ax.plot(plt_idx, offline_td3_df.loc[plt_idx, col], label=f'[offline td3] {col}')
        # if col == 'q1_repr_nonzero':
        #     ax.plot(plt_idx, td3_df.loc[plt_idx, 'q2_repr_nonzero'], label=f'[td3] q2_repr_nonzero')
        #     ax.plot(plt_idx, td3bc_df.loc[plt_idx, 'q2_repr_nonzero'], label=f'[td3bc] q2_repr_nonzero')
        #     ax.plot(plt_idx, offline_td3_df.loc[plt_idx, 'q2_repr_nonzero'], label=f'[offline td3] q2_repr_nonzero')
        if col == 'normalized_dot_prod_repr_q1':
            ax.set_yscale('log')
            ax.plot(plt_idx, td3_df.loc[plt_idx, 'normalized_dot_prod_repr_q2'], label=f'[td3] normalized_dot_prod_repr_q2')
            ax.plot(plt_idx, td3bc_df.loc[plt_idx, 'normalized_dot_prod_repr_q2'], label=f'[td3bc] normalized_dot_prod_repr_q2')
            ax.plot(plt_idx, offline_td3_df.loc[plt_idx, 'normalized_dot_prod_repr_q2'], label=f'[offline td3] normalized_dot_prod_repr_q2')
        if col == 'q1_repr_norm':
            ax.set_yscale('log')
            ax.plot(plt_idx, td3_df.loc[plt_idx, 'q2_repr_norm'], label=f'[td3] q2_repr_norm')
            ax.plot(plt_idx, td3bc_df.loc[plt_idx, 'q2_repr_norm'], label=f'[td3bc] q2_repr_norm')
            ax.plot(plt_idx, offline_td3_df.loc[plt_idx, 'q2_repr_norm'], label=f'[offline td3] q2_repr_norm')
        if col == 'q1_kernel_norm':
            ax.plot(plt_idx, td3_df.loc[plt_idx, 'q2_kernel_norm'], label=f'[td3] q2_kernel_norm')
            ax.plot(plt_idx, td3bc_df.loc[plt_idx, 'q2_kernel_norm'], label=f'[td3bc] q2_kernel_norm')
            ax.plot(plt_idx, offline_td3_df.loc[plt_idx, 'q2_kernel_norm'], label=f'[offline td3] q2_kernel_norm')
        if col == 'q1_output_norm':
            ax.plot(plt_idx, td3_df.loc[plt_idx, 'q2_output_norm'], label=f'[td3] q2_output_norm')
            ax.plot(plt_idx, td3bc_df.loc[plt_idx, 'q2_output_norm'], label=f'[td3bc] q2_output_norm')
            ax.plot(plt_idx, offline_td3_df.loc[plt_idx, 'q2_output_norm'], label=f'[offline td3] q2_output_norm')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{env_name}.png', dpi=360)


if __name__ == "__main__":
    plot_repr_similarity()
