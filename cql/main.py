import d4rl
import gym
import jax
import os
import time
import numpy as np
import pandas as pd
from tqdm import trange

from models import CQLAgent
from utils import ReplayBuffer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"


def eval_policy(agent: CQLAgent,
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
                                                   eval_rng, np.array(obs), True)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr_actor", default=1e-4, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--eval_freq", default=int(5e3), type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--target_entropy", default=None, type=float)
    parser.add_argument("--auto_entropy_tuning", default=True, action="store_false")
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--model_dir", default="./saved_models", type=str)
    parser.add_argument("--with_lagrange", default=False, action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    # Env parameters
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # max_action = float(env.action_space.high[0])

    # random seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)

    # TD3 agent
    agent = CQLAgent(obs_dim=obs_dim,
                     act_dim=act_dim,
                     seed=args.seed,
                     tau=args.tau,
                     gamma=args.gamma,
                     lr=args.lr,
                     lr_actor=args.lr_actor,
                     auto_entropy_tuning=args.auto_entropy_tuning,
                     target_entropy=args.target_entropy,
                     with_lagrange=args.with_lagrange)

    # Replay D4RL buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    # Evaluate the untrained policy
    logs = [{"step": 0, "reward": eval_policy(agent, args.env, args.seed)}]

    # Initialize training stats
    start_time = time.time()

    # Train agent and evaluate policy
    for t in trange(args.max_timesteps):
        log_info = agent.update(replay_buffer, args.batch_size)

        if (t + 1) % args.eval_freq == 0:
            eval_reward = eval_policy(agent, args.env, args.seed)
            log_info.update({
                "step": t+1,
                "reward": eval_reward,
                "time": (time.time() - start_time) / 60
            })
            logs.append(log_info)
            if args.with_lagrange:
                print(
                    f"# Step {t+1}: {eval_reward:.2f}, critic_loss: {log_info['critic_loss']:.3f}, "
                    f"actor_loss: {log_info['actor_loss']:.3f}, alpha_loss: {log_info['alpha_loss']:.3f}, "
                    f"alpha: {log_info['alpha']:.3f}, cql_alpha: {log_info['cql_alpha']:.3f}, "
                    f"q1: {log_info['q1']:.3f}, q2: {log_info['q2']:.3f}, "
                    f"ood_q1: {log_info['ood_q1']:.3f}, ood_q2: {log_info['ood_q2']:.3f}"
                )
            else:
                print(
                    f"# Step {t+1}: {eval_reward:.2f}, critic_loss: {log_info['critic_loss']:.3f}, "
                    f"actor_loss: {log_info['actor_loss']:.3f}, alpha_loss: {log_info['alpha_loss']:.3f}, "
                    f"alpha: {log_info['alpha']:.3f}, "
                    f"q1: {log_info['q1']:.3f}, q2: {log_info['q2']:.3f}, "
                    f"random_q1: {log_info['q1_random']:.3f}, random_q2: {log_info['q2_random']:.3f}"
                )

    # Save logs
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/{args.env}", exist_ok=True)
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{args.log_dir}/{args.env}/{args.seed}.csv")
    # agent.save(f"{args.model_dir}/{args.env}/{args.seed}")


if __name__ == "__main__":
    args = get_args()
    print(f"\nArguments:\n{vars(args)}")
    main(args)
