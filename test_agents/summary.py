from tqdm import tqdm
import pandas as pd

env_name = 'hopper-medium-v2'


for target in ['reward', 'next_obs', 'action']:
    res = []
    for algo in ['combo', 'cql', 'td3bc', 'random']:
        df = pd.read_csv(f'res/{env_name}/{algo}_probe_{target}.csv', index_col=0)
        res.append((algo, df['cv_loss'].mean(), df['cv_loss'].std()))
    res_df = pd.DataFrame(res, columns=['algo', 'mean', 'std'])
    res_df.to_csv(f'res/{target}.csv')


def summarize_probe_res_mu_std():
    res = []
    df = pd.read_csv('probe_exp_res/action.csv', index_col=0)
    env_names = df['env_name'].unique()
    algo_names = df['algo'].unique()
    for env_name in env_names:
        for algo in algo_names:
            tmp_df = df.query(f'env_name == "{env_name}" & algo == "{algo}"')
            cv_losses = tmp_df.loc[:, [f'cv_loss{i}' for i in range(1, 6)]].values.reshape(-1)
            assert cv_losses.shape == (50,)
            res.append((env_name, algo, cv_losses.mean(), cv_losses.std()))
    res_df = pd.DataFrame(res, columns=['env_name', 'algo', 'mu', 'std'])
    res_df.to_csv('probe_exp_res/action_mu_std.csv')


def summarize_repr_rank_res():
    baseline_agent_df = pd.read_csv('config/baseline_agent.csv', index_col=0)
    baseline_agent_df = baseline_agent_df.set_index(['env_name', 'algo', 'seed'])
    res_df = pd.read_csv('probe_exp_res/repr_rank_res.csv', index_col=0)
    sigma_cols = [f'sigma{i}' for i in range(1, 257)]
    res_df['collased_dimensions'] = (res_df.loc[:, sigma_cols] < 1e-6).sum(1)
    res_df['max_sigma'] = res_df.loc[:, sigma_cols].max(1)
    res_df['min_sigma'] = res_df.loc[:, sigma_cols].min(1)

    rewards_dict = {}
    for algo in ['combo', 'cql', 'td3bc']:
        rewards_dict[algo] = {}
        for env_name in ['hopper-medium-v2', 'walker2d-medium-expert-v2', 'halfcheetah-medium-replay-v2']:
            df = pd.read_csv(f'eval_agent_res/{algo}/{env_name}_agent.csv', index_col=0)
            rewards_dict[algo][env_name] = df.set_index(['env', 'seed', 'step'])

    new_df = res_df.loc[:, ['env_name', 'algo', 'seed', 'size', 'zero_col_num', 'embedding',
                            'collased_dimensions', 'max_sigma', 'min_sigma']].copy()
    eval_rewards = []
    for i in tqdm(new_df.index):
        env_name, algo, seed = new_df.loc[i, ['env_name', 'algo', 'seed']]
        step = baseline_agent_df.loc[(env_name, algo, seed), 'step']
        eval_reward = rewards_dict[algo][env_name].loc[(env_name, seed, step), 'reward']
        eval_rewards.append(eval_reward)
    new_df['eval_reward'] = eval_rewards

    actor_embedding_df = new_df.query('embedding == "actor_embedding"')
    critic_embedding_df = new_df.query('embedding == "critic_embedding"')

    actor_df1 = actor_embedding_df.query('size==10000')
    actor_df3 = actor_embedding_df.query('size==30000')
    actor_df5 = actor_embedding_df.query('size==50000')

    critic_df1 = critic_embedding_df.query('size==10000')
    critic_df3 = critic_embedding_df.query('size==30000')
    critic_df5 = critic_embedding_df.query('size==50000')

    halfcheetah_critic_df = critic_df5.query('env_name=="halfcheetah-medium-replay-v2"').loc[:, ['algo', 'seed', 'zero_col_num', 'collased_dimensions', 'max_sigma', 'min_sigma', 'eval_reward']]
    
    hopper_critic_df = critic_df5.query('env_name=="hopper-medium-v2"').loc[:, ['algo', 'seed', 'zero_col_num', 'collased_dimensions', 'max_sigma', 'min_sigma', 'eval_reward']]

    walker2d_critic_df = critic_df5.query('env_name=="walker2d-medium-expert-v2"').loc[:, ['algo', 'seed', 'zero_col_num', 'collased_dimensions', 'max_sigma', 'min_sigma', 'eval_reward']]

    walker2d_actor_df = actor_df5.query('env_name=="walker2d-medium-expert-v2"').loc[:, ['algo', 'seed', 'zero_col_num', 'collased_dimensions', 'max_sigma', 'min_sigma', 'eval_reward']]

