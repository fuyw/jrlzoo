"""[5, 6, 4, 0, 2]"""
import d4rl
import gym
import jax
import logging
import os
import time
import numpy as np
import pandas as pd
from tqdm import trange

from models import COMBOAgent
from utils import Batch, ReplayBuffer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"


def eval_policy(agent: COMBOAgent, env_name: str, seed: int, eval_episodes: int = 10) -> float:
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    eval_rng = jax.random.PRNGKey(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        t = 0
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            eval_rng, action = agent.select_action(agent.actor_state.params, eval_rng, np.array(obs), True)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return avg_reward
    # return d4rl_score


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-v0")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr_actor", default=3e-5, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--eval_freq", default=int(5e3), type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--target_entropy", default=None, type=float)
    parser.add_argument("--min_q_weight", default=3.0, type=float)
    parser.add_argument("--auto_entropy_tuning", default=True, action="store_false")
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--model_dir", default="./ensemble_models", type=str)
    parser.add_argument("--backup_entropy", default=False, action="store_true")
    parser.add_argument("--pool", default=False, action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    exp_name = f'combo_s{args.seed}_alpha{args.min_q_weight}_check'
    exp_info = f'# Running experiment for: {exp_name} #'
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    # Log setting
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f'{args.log_dir}/{args.env}/{exp_name}.log',
                        filemode='w',
                        force=True)
    logger = logging.getLogger()

    # Env parameters
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # random seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)

    # TD3 agent
    agent = COMBOAgent(env=args.env, obs_dim=obs_dim, act_dim=act_dim, seed=args.seed,
                       lr=args.lr, lr_actor=args.lr_actor, rollout_batch_size=10000)
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    # Initialize training stats
    start_time = time.time()

    M = 800
    check_data = np.load(f'/usr/local/data/yuweifu/jaxrl/benchmarks/check_hopper_{M}k.npz')
    check_obs = check_data['obs']
    check_act = check_data['act']
    check_rew = check_data['rew']
    check_done = check_data['done']
    check_next_obs = check_data['next_obs']
    print(f'Take {(time.time() - start_time)/60:.2f} min to load the data')

    # Evaluate the untrained policy
    logs = [{"step": 0, "reward": eval_policy(agent, args.env, args.seed)}]  # 2.382196339051178

    # Train agent and evaluate policy
    # for t in trange(M*1000):
    #     real_batch = Batch(
    #         observations=check_obs[t][:128],
    #         actions=check_act[t][:128],
    #         rewards=check_rew[t][:128],
    #         discounts= 1 - check_done[t][:128],
    #         next_observations=check_next_obs[t][:128]
    #     )
        real_batch = replay_buffer.sample(128)
        model_batch = Batch(
            observations=check_obs[t][128:],
            actions=check_act[t][128:],
            rewards=check_rew[t][128:],
            discounts= 1 - check_done[t][128:],
            next_observations=check_next_obs[t][128:]
        )
        log_info = agent.update_check(real_batch, model_batch)
        if (t + 1) % args.eval_freq == 0:
            eval_reward = eval_policy(agent, args.env, args.seed)
            log_info.update({
                "step": t+1,
                "reward": eval_reward,
                "time": (time.time() - start_time) / 60
            })
            logs.append(log_info)
            logger.info(
                f"\n# Step {t+1}: eval_reward = {eval_reward:.2f}, time: {log_info['time']:.2f}\n"
                f"\talpha_loss: {log_info['alpha_loss']:.2f}, alpha: {log_info['alpha']:.2f}, logp: {log_info['logp']:.2f}\n"
                f"\tactor_loss: {log_info['actor_loss']:.2f}, sampled_q: {log_info['sampled_q']:.2f}\n"

                f"\tQ_loss_1: {log_info['Q_loss_1']:.2f}, Q_loss_2: {log_info['Q_loss_2']:.2f}\n"

                f"\tcritic_loss: {log_info['critic_loss']:.2f}, critic_loss_min: {log_info['critic_loss_min']:.2f}, "
                f"critic_loss_max: {log_info['critic_loss_max']:.2f}, critic_loss_std: {log_info['critic_loss_std']:.2f}\n"

                f"\tcritic_loss1: {log_info['critic_loss1']:.2f}, critic_loss1_min: {log_info['critic_loss1_min']:.2f}, "
                f"critic_loss1_max: {log_info['critic_loss1_max']:.2f}, critic_loss1_std: {log_info['critic_loss1_std']:.2f}\n"

                f"\tcritic_loss2: {log_info['critic_loss2']:.2f}, critic_loss2_min: {log_info['critic_loss2_min']:.2f}, "
                f"critic_loss2_max: {log_info['critic_loss2_max']:.2f}, critic_loss2_std: {log_info['critic_loss2_std']:.2f}\n"

                f"\tcql1_loss: {log_info['cql1_loss']:.2f}, cql1_loss_min: {log_info['cql1_loss_min']:.2f} "
                f"cql1_loss_max: {log_info['cql1_loss_max']:.2f}, cql1_loss_std: {log_info['cql1_loss_std']:.2f}\n"

                f"\tcql2_loss: {log_info['cql2_loss']:.2f}, cql2_loss_min: {log_info['cql2_loss_min']:.2f} "
                f"cql2_loss_max: {log_info['cql2_loss_max']:.2f}, cql2_loss_std: {log_info['cql2_loss_std']:.2f}\n"

                f"\ttarget_q: {log_info['target_q']:.2f}, target_q_min: {log_info['target_q_min']:.2f} "
                f"target_q_max: {log_info['target_q_max']:.2f}, target_q_std: {log_info['target_q_std']:.2f}\n"

                f"\tq1: {log_info['q1']:.2f}, q1_min: {log_info['q1_min']:.2f}, q1_max: {log_info['q1_max']:.2f}, q1_std: {log_info['q1_std']:.2f}\n"

                f"\tq2: {log_info['q2']:.2f}, q2_min: {log_info['q2_min']:.2f}, q2_max: {log_info['q2_max']:.2f}, q2_std: {log_info['q2_std']:.2f}\n"

                f"\tood_q1: {log_info['ood_q1']:.2f}, ood_q1_min: {log_info['ood_q1_min']:.2f}, "
                f"ood_q1_max: {log_info['ood_q1_max']:.2f}, ood_q1_std: {log_info['ood_q1_std']:.2f}\n"

                f"\tood_q2: {log_info['ood_q2']:.2f}, ood_q2_min: {log_info['ood_q2_min']:.2f}, "
                f"ood_q2_max: {log_info['ood_q2_max']:.2f}, ood_q2_std: {log_info['ood_q2_std']:.2f}\n"

                f"\tcql_q1: {log_info['cql_q1']:.2f}, cql_q2: {log_info['cql_q2']:.2f}\n"
                f"\trandom_q1: {log_info['random_q1']:.2f}, random_q2: {log_info['random_q2']:.2f}, "
                f"logp_next_action: {log_info['logp_next_action']:.2f}\n"

                f"\treal_batch_rewards: {log_info['real_batch_rewards']:.2f}, real_batch_obs: {log_info['real_batch_obs']:.2f}, real_batch_actions: {log_info['real_batch_actions']:.2f}, real_batch_dones: {log_info['real_batch_dones']:.2f}\n"
                f"\tmodel_batch_rewards: {log_info['model_batch_rewards']:.2f}, model_batch_obs: {log_info['model_batch_obs']:.2f}, model_batch_actions: {log_info['model_batch_actions']:.2f}, model_batch_dones: {log_info['model_batch_dones']:.2f}\n"
            )

    # Save logs
    log_name = f"s{args.seed}"
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{args.log_dir}/{args.env}/{exp_name}.csv")
    # agent.save(f"{args.model_dir}/{args.env}/{args.seed}")


if __name__ == "__main__":
    args = get_args()
    print(f"\nArguments:\n{vars(args)}")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/{args.env}", exist_ok=True)
    main(args)
