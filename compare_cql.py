"""
Index(['step', 'reward', 'actor_loss', 'alpha', 'alpha_loss', 'cql1_loss',
       'cql2_loss', 'cql_next_q1', 'cql_next_q2', 'cql_q1', 'cql_q2',
       'critic_loss', 'logp', 'logp_next_action', 'q1', 'q2', 'random_q1',
       'random_q2', 'target_q', 'time'],

Index(['step', 'reward', 'actor_loss', 'alpha', 'alpha_loss', 'cql_alpha',
       'cql_diff1', 'cql_diff2', 'cql_loss1', 'cql_loss1_max', 'cql_loss1_min',
       'cql_loss1_std', 'cql_loss2', 'cql_loss2_max', 'cql_loss2_min',
       'cql_loss2_std', 'cql_q1', 'cql_q2', 'critic_loss', 'critic_loss1',
       'critic_loss1_max', 'critic_loss1_min', 'critic_loss1_std',
       'critic_loss2', 'critic_loss2_max', 'critic_loss2_min',
       'critic_loss2_std', 'critic_loss_max', 'critic_loss_min',
       'critic_loss_std', 'logp', 'logp_next_action', 'min_q_weight', 'ood_q1',
       'ood_q1_max', 'ood_q1_min', 'ood_q1_std', 'ood_q2', 'ood_q2_max',
       'ood_q2_min', 'ood_q2_std', 'q1', 'q1_max', 'q1_min', 'q1_std', 'q2',
       'q2_max', 'q2_min', 'q2_std', 'random_q1', 'random_q2', 'sampled_q',
       'target_q', 'target_q_max', 'target_q_min', 'target_q_std', 'eval_time',
       'time'],
      dtype='object')
"""
import pandas as pd
import matplotlib.pyplot as plt
error_df = pd.read_csv('/usr/local/data/yuweifu/jaxrl/case_study/cql/tmp/logs/halfcheetah-medium-expert-v2/cql_s0.csv', index_col=0).set_index('step')
right_df = pd.read_csv('/usr/local/data/yuweifu/jaxrl/cql/logs/halfcheetah-medium-expert-v2/0.csv', index_col=0).set_index('step')

_, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
plt_idx = range(0, 1010000, 10000)
axes[0].plot(plt_idx, error_df.loc[plt_idx, "reward"], label="error")
axes[0].plot(plt_idx, right_df.loc[plt_idx, "reward"], label="right")
axes[0].set_title("reward")
axes[1].plot(plt_idx, error_df.loc[plt_idx, "cql_loss1"], label="error")
axes[1].plot(plt_idx, right_df.loc[plt_idx, "cql1_loss"], label="right")
axes[1].set_title("cql_loss1")
axes[2].plot(plt_idx, error_df.loc[plt_idx, "q1"], label="error")
axes[2].plot(plt_idx, right_df.loc[plt_idx, "q1"], label="right")
axes[2].set_title("q1")
axes[3].plot(plt_idx, error_df.loc[plt_idx, "critic_loss"], label="error")
axes[3].plot(plt_idx, right_df.loc[plt_idx, "critic_loss"], label="right")
axes[3].set_title("critic_loss")
axes[4].plot(plt_idx, error_df.loc[plt_idx, "alpha"], label="error")
axes[4].plot(plt_idx, right_df.loc[plt_idx, "alpha"], label="right")
axes[4].set_title("alpha")
plt.tight_layout()
plt.savefig("compare.png", dpi=540)
