"""
# Step 115000: eval_reward = 5.87
        alpha_loss: 0.04, alpha: 1.10, logp: 2.53
        actor_loss: -21.19, sampled_q: 23.96
        critic_loss: 9.45, q1: 25.46, q2: 25.31, target_q: 25.56
        cql1_loss: -1.19, cql2_loss: -1.16
        cql_q1: 24.67, cql_next_q1: 24.07, random_q1: 13.91 
        cql_q2: 24.51, cql_next_q2: 23.96, random_q2: 14.13 
        logp_next_action: 2.65, cql_logp: 2.77 cql_logp_next_action: 2.76
        batch_rewards: 1.70, batch_discounts: 0.98, batch_obs: 7.76, buffer_size: 115000
        fix_q1: -57.81, fix_q2: -38.21
        Episode 2247: steps = 80, reward = 156.88
"""
import pandas as pd


def convert_log(input_name, output_name):
    df = pd.read_csv(input_name, index_col=0)
    with open(output_name, 'w') as f:
        for i in range(1, len(df)):
            (step, reward, alpha_loss, alpha, logp, actor_loss, sampled_q,
             critic_loss, q1, q2, target_q, cql1_loss, cql2_loss, cql_q1,
             cql_next_q1, random_q1, cql_q2, cql_next_q2, random_q2,
             logp_next_action, cql_logp, cql_logp_next_action) = df.iloc[i][[
                 'step', 'reward', 'alpha_loss', 'alpha', 'logp', 'actor_loss',
                 'sampled_q', 'critic_loss', 'q1', 'q2', 'target_q',
                 'cql1_loss', 'cql2_loss', 'cql_q1', 'cql_next_q1',
                 'random_q1', 'cql_q2', 'cql_next_q2', 'random_q2',
                 'logp_next_action', 'cql_logp', 'cql_logp_next_action'
             ]]
            step = int(step)
            if step % 5000 == 0:
                f.write(
                    f'# Step {step}: eval_reward = {reward:.2f}\n'
                    f'\talpha_loss: {alpha_loss:.2f}, alpha: {alpha:.2f}, logp: {logp:.2f}\n'
                    f'\tactor_loss: {actor_loss:.2f}, sampled_q: {sampled_q:.2f}\n'
                    f'\tcritic_loss: {critic_loss:.2f}, q1: {q1:.2f}, q2: {q2:.2f}, target_q: {target_q:.2f}\n'
                    f'\tcql1_loss: {cql1_loss:.2f}, cql2_loss: {cql2_loss:.2f}\n'
                    f'\tcql_q1: {cql_q1:.2f}, cql_next_q1: {cql_next_q1:.2f}, random_q1: {random_q1:.2f}\n'
                    f'\tcql_q2: {cql_q2:.2f}, cql_next_q2: {cql_next_q2:.2f}, random_q2: {random_q1:.2f}\n'
                    f'\tlogp_next_action: {logp_next_action:.2f}, cql_logp: {cql_logp:.2f}, cql_logp_next_action: {cql_logp_next_action:.2f}\n\n'
                )


if __name__ == '__main__':
    input_name = 'logs/hopper-medium-v2/td3_buffer_s0.csv'
    output_name = 'logs/hopper-medium-v2/td3_buffer_cql_s0.log'
    convert_log(input_name, output_name)
