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
from utils import ReplayBuffer


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"


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
    return d4rl_score


conf_dict = {
    "walker2d-medium-v2": {"lr_actor": 1e-5, "lr": 1e-4, "min_q_weight": 3.0, "horizon": 1, "rollout_random": False},
    "walker2d-medium-replay-v2": {"lr_actor": 1e-5, "lr": 1e-4, "min_q_weight": 1.0, "horizon": 1, "rollout_random": False},
    "walker2d-medium-expert-v2": {"lr_actor": 1e-5, "lr": 1e-4, "min_q_weight": 3.0, "horizon": 1, "rollout_random": False},
    "hopper-medium-v2": {"lr_actor": 1e-4, "lr": 3e-4, "min_q_weight": 3.0, "horizon": 5, "rollout_random": False},
    "hopper-medium-replay-v2": {"lr_actor": 1e-4, "lr": 3e-4, "min_q_weight": 1.0, "horizon": 5, "rollout_random": True},
    "hopper-medium-expert-v2": {"lr_actor": 1e-5, "lr": 1e-4, "min_q_weight": 3.0, "horizon": 3, "rollout_random": False},
    "halfcheetah-medium-v2": {"lr_actor": 1e-5, "lr": 1e-4, "min_q_weight": 1.0, "horizon": 5, "rollout_random": False},
    "halfcheetah-medium-replay-v2": {"lr_actor": 1e-4, "lr": 3e-4, "min_q_weight": 1.0, "horizon": 5, "rollout_random": False},
    "halfcheetah-medium-expert-v2": {"lr_actor": 1e-5, "lr": 1e-4, "min_q_weight": 5.0, "horizon": 5, "rollout_random": False},
}


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="walker2d-medium-replay-v2")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr_actor", default=1e-4, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--min_q_weight", default=5.0, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--eval_freq", default=int(5e3), type=int)
    parser.add_argument("--horizon", default=1, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--target_entropy", default=None, type=float)
    parser.add_argument("--auto_entropy_tuning", default=True, action="store_false")
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--model_dir", default="./saved_dynamics_models", type=str)
    parser.add_argument("--combo_dir", default="./saved_combo_models", type=str)
    parser.add_argument("--backup_entropy", default=False, action="store_true")
    parser.add_argument("--rollout_random", default=False, action="store_true")
    parser.add_argument("--train_dynamics_model", default=False, action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    exp_name = f'combo_s{args.seed}_alpha{args.min_q_weight}_h{args.horizon}'
    exp_info = f'# Running experiment for: {exp_name}_{args.env_name} #'
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    # Log setting
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f'{args.log_dir}/{args.env_name}/{exp_name}.log',
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    logger.info(f"\nArguments:\n{vars(args)}")

    # Env parameters
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # random seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)

    # COMBO agent
    agent = COMBOAgent(env_name=args.env_name,
                       obs_dim=obs_dim,
                       act_dim=act_dim,
                       seed=args.seed,
                       lr=args.lr,
                       lr_actor=args.lr_actor,
                       horizon=args.horizon,
                       rollout_batch_size=10000,
                       min_q_weight=args.min_q_weight,
                       rollout_random=args.rollout_random)

    # Train the dynamics model
    if args.train_dynamics_model:
        agent.model.train()

    # Load the trained dynamics model
    agent.model.load(f'{args.model_dir}/{args.env_name}')

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    model_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(args.horizon*1e5))

    # Evaluate the untrained policy
    logs = [{"step": 0, "reward": eval_policy(agent, args.env_name, args.seed)}]  # 2.38219

    # Initialize training stats
    start_time = time.time()

    # Train agent and evaluate policy
    for t in trange(args.max_timesteps):
        log_info = agent.update(replay_buffer, model_buffer)
        # log_info = agent.update2(replay_buffer, model_buffer)

        # save some evaluate time
        if ((t + 1) >= int(9.5e5) and
            (t + 1) % args.eval_freq == 0) or ((t + 1) <= int(9.5e5) and
                                               (t + 1) %
                                               (2 * args.eval_freq) == 0):
            eval_reward = eval_policy(agent, args.env_name, args.seed)
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

                f"\tcql_q1: {log_info['cql_q1']:.2f}, cql_q2: {log_info['cql_q2']:.2f}\n"
                f"\trandom_q1: {log_info['random_q1']:.2f}, random_q2: {log_info['random_q2']:.2f}, "
                f"logp_next_action: {log_info['logp_next_action']:.2f}\n"

                f"\treal_batch_rewards: {log_info['real_batch_rewards']:.2f}, "
                f"real_batch_actions: {log_info['real_batch_actions']:.2f}, "
                f"real_batch_discounts: {log_info['real_batch_discounts']:.2f}\n"
                f"\tmodel_batch_rewards: {log_info['model_batch_rewards']:.2f}, "
                f"model_batch_actions: {log_info['model_batch_actions']:.2f}, "
                f"model_batch_discounts: {log_info['model_batch_discounts']:.2f}\n"
                f"\tmodel_buffer_size: {log_info['model_buffer_size']:.0f}, "
                f"model_buffer_ptr: {log_info['model_buffer_ptr']:.0f}\n"
                f"\tmin_q_weight: {log_info['min_q_weight']:.1f}\n"
            )

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{args.log_dir}/{args.env_name}/{exp_name}.csv")
    agent.save(f"{args.combo_dir}/{args.env_name}/s{args.seed}")


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.combo_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env_name}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/{args.env_name}", exist_ok=True)
    os.makedirs(f"{args.combo_dir}/{args.env_name}", exist_ok=True)

    # set parameters according to envs
    args.lr_actor = conf_dict[args.env_name]["lr_actor"]
    args.lr = conf_dict[args.env_name]["lr"]
    args.min_q_weight = conf_dict[args.env_name]["min_q_weight"]
    args.horizon = conf_dict[args.env_name]["horizon"]
    args.rollout_random = conf_dict[args.env_name]["rollout_random"]
    main(args)
