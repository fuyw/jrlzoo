import d4rl
import gym
import jax
import json
import logging
import os
import time
import numpy as np
import pandas as pd
from tqdm import trange

from models import COMBOAgent
from utils import ReplayBuffer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"


def eval_policy(agent: COMBOAgent,
                env_name: str,
                seed: int,
                eval_episodes: int = 10) -> float:
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    eval_rng = jax.random.PRNGKey(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        t = 0
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            eval_rng, action = agent.select_action(agent.actor_state.params,
                                                   eval_rng, np.array(obs),
                                                   True)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    # parser.add_argument("--env", default="walker2d-medium-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hid_dim", default=256, type=int)
    parser.add_argument("--hid_layers", default=3, type=int)
    parser.add_argument("--lr_actor", default=1e-4, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--max_timesteps", default=int(4e5), type=int)
    parser.add_argument("--eval_freq", default=int(5e3), type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--min_q_weight", default=3.0, type=float)
    parser.add_argument("--target_entropy", default=None, type=float)
    parser.add_argument("--auto_entropy_tuning",
                        default=True,
                        action="store_false")
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--model_dir", default="./saved_models", type=str)
    parser.add_argument("--backup_entropy", default=False, action="store_true")
    parser.add_argument("--with_lagrange", default=False, action="store_true")
    parser.add_argument("--cql", default=False, action="store_true")
    parser.add_argument("--name", default="combo1", type=str)
    parser.add_argument("--lagrange_thresh", default=5.0, type=float)
    parser.add_argument("--real_ratio", default=0.5, type=float)
    args = parser.parse_args()
    return args


def main(args):
    exp_name = f'{args.name}_s{args.seed}_alpha{args.min_q_weight}_rr{args.real_ratio}'
    exp_info = f'# Running experiment for: {exp_name}_{args.env} #'
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    # Log setting
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f'logs/{args.env}/{exp_name}.log',
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    logger.info(f"\nArguments:\n{vars(args)}")

    # Env parameters
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # random seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    np.random.seed(args.seed)

    # CQL agent
    agent = COMBOAgent(obs_dim=obs_dim,
                       act_dim=act_dim,
                       hid_dim=args.hid_dim,
                       env=args.env,
                       hid_layers=args.hid_layers,
                       seed=args.seed,
                       tau=args.tau,
                       gamma=args.gamma,
                       lr=args.lr,
                       lr_actor=args.lr_actor,
                       auto_entropy_tuning=args.auto_entropy_tuning,
                       backup_entropy=args.backup_entropy,
                       target_entropy=args.target_entropy,
                       min_q_weight=args.min_q_weight,
                       with_lagrange=args.with_lagrange,
                       lagrange_thresh=args.lagrange_thresh,
                       real_ratio=args.real_ratio)

    # logger.info(f"\nThe actor architecture is:\n{jax.tree_map(lambda x: x.shape, agent.actor_state.params)}")
    # logger.info(f"\nThe critic architecture is:\n{jax.tree_map(lambda x: x.shape, agent.critic_state.params)}")

    # Replay D4RL buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    model_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(5e5))
    fix_obs = np.random.normal(size=(128, obs_dim))
    fix_act = np.random.normal(size=(128, act_dim))

    # Evaluate the untrained policy
    logs = [{"step": 0, "reward": eval_policy(agent, args.env, args.seed)}]

    # Initialize training stats
    start_time = time.time()

    # Train agent and evaluate policy
    for t in trange(args.max_timesteps):
        log_info = agent.update(replay_buffer, model_buffer)

        # save some evaluate time
        if ((t + 1) >= int(9.5e5) and
            (t + 1) % args.eval_freq == 0) or ((t + 1) <= int(9.5e5) and
                                               (t + 1) %
                                               (2 * args.eval_freq) == 0):
            # eval_reward = np.NaN
            eval_reward = eval_policy(agent, args.env, args.seed)
            log_info.update({
                "step": t + 1,
                "reward": eval_reward,
                "time": (time.time() - start_time) / 60
            })
            logs.append(log_info)
            fix_q1, fix_q2 = agent.critic.apply(
                {"params": agent.critic_state.params}, fix_obs, fix_act)
            _, fix_a = agent.select_action(agent.actor_state.params,
                                           jax.random.PRNGKey(0), fix_obs,
                                           True)
            logger.info(
                f"\n# Step {t+1}: eval_reward = {eval_reward:.2f}, time: {log_info['time']:.2f}\n"
                f"\talpha_loss: {log_info['alpha_loss']:.2f}, alpha: {log_info['alpha']:.2f}, logp: {log_info['logp']:.2f}\n"
                f"\tactor_loss: {log_info['actor_loss']:.2f}, sampled_q: {log_info['sampled_q']:.2f}\n"
                f"\tcritic_loss: {log_info['critic_loss']:.2f}, critic_loss_min: {log_info['critic_loss_min']:.2f}, "
                f"critic_loss_max: {log_info['critic_loss_max']:.2f}, critic_loss_std: {log_info['critic_loss_std']:.2f}\n"
                f"\tcritic_loss1: {log_info['critic_loss1']:.2f}, critic_loss1_min: {log_info['critic_loss1_min']:.2f}, "
                f"critic_loss1_max: {log_info['critic_loss1_max']:.2f}, critic_loss1_std: {log_info['critic_loss1_std']:.2f}\n"
                f"\tcritic_loss2: {log_info['critic_loss2']:.2f}, critic_loss2_min: {log_info['critic_loss2_min']:.2f}, "
                f"critic_loss2_max: {log_info['critic_loss2_max']:.2f}, critic_loss2_std: {log_info['critic_loss2_std']:.2f}\n"
                f"\treal_critic_loss: {log_info['real_critic_loss']:.2f}, fake_critic_loss: {log_info['fake_critic_loss']:.2f}\n"
                f"\treal_critic_loss_max: {log_info['real_critic_loss_max']:.2f}, fake_critic_loss_min: {log_info['fake_critic_loss_min']:.2f}\n"
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
                f"\tcql_q1_avg: {log_info['cql_q1_avg']:.2f}, cql_q1_min: {log_info['cql_q1_min']:.2f}, cql_q1_max: {log_info['cql_q1_max']:.2f}\n"
                f"\tcql_q2_avg: {log_info['cql_q2_avg']:.2f}, cql_q2_min: {log_info['cql_q2_min']:.2f}, cql_q2_max: {log_info['cql_q2_max']:.2f}\n"
                f"\trandom_q1_avg: {log_info['random_q1_avg']:.2f}, random_q1_min: {log_info['random_q1_min']:.2f}, random_q1_max: {log_info['random_q1_max']:.2f}\n"
                f"\trandom_q2_avg: {log_info['random_q2_avg']:.2f}, random_q2_min: {log_info['random_q2_min']:.2f}, random_q2_max: {log_info['random_q2_max']:.2f}\n"
                f"\tlogp_next_action: {log_info['logp_next_action']:.2f}, cql_logp: {log_info['cql_logp']:.2f} \n"
                # f"cql_logp_next_action: {log_info['cql_logp_next_action']:.2f}\n"

                f"\treal_batch_rewards: {log_info['real_batch_rewards']:.2f}, real_batch_obs: {log_info['real_batch_obs']:.2f}, real_batch_actions: {log_info['real_batch_actions']:.2f}, real_batch_dones: {log_info['real_batch_dones']:.2f}\n"

                f"\tmodel_batch_rewards: {log_info['model_batch_rewards']:.2f}, model_batch_obs: {log_info['model_batch_obs']:.2f}, model_batch_actions: {log_info['model_batch_actions']:.2f}, model_batch_dones: {log_info['model_batch_dones']:.2f}\n"
                f"\tmodel_buffer_size: {log_info['model_buffer_size']:.0f}, model_buffer_ptr: {log_info['model_buffer_ptr']:.0f}\n"
                f"\treal_batch_size: {log_info['real_batch_size']:.0f}, fake_batch_size: {log_info['fake_batch_size']:.0f}"
                f"\tfix_q1: {fix_q1.squeeze().mean().item():.2f}, fix_q2: {fix_q2.squeeze().mean().item():.2f}, fix_a: {abs(fix_a).sum().item():.2f}\n\n"
            )

            log_df = pd.DataFrame(logs)
            log_df.to_csv(f"{args.log_dir}/{args.env}/{exp_name}.csv")
    with open(f"{args.log_dir}/{args.env}/{exp_name}.json", "w") as f:
        json.dump(vars(args), f)


if __name__ == "__main__":
    args = get_args()
    print(f"\nArguments:\n{vars(args)}")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/{args.env}", exist_ok=True)
    main(args)
