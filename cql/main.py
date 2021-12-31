import d4rl
import gym
import jax
import json
import os
import time
import numpy as np
import pandas as pd
from tqdm import trange

from models import CQLAgent
from utils import ReplayBuffer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"


def eval_policy(agent: CQLAgent, env_name: str, seed: int, eval_episodes: int = 10) -> float:
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


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hid_dim", default=256, type=int)
    parser.add_argument("--hid_layer", default=3, type=int)
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
    parser.add_argument("--backup_entropy", default=False, action="store_true")
    parser.add_argument("--with_lagrange", default=False, action="store_true")
    parser.add_argument("--lagrange_thresh", default=5.0, type=float)
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
    env.observation_space.seed(args.seed)
    np.random.seed(args.seed)

    # CQL agent
    agent = CQLAgent(obs_dim=obs_dim,
                     act_dim=act_dim,
                     hid_dim=args.hid_dim,
                     hid_layer=args.hid_layer,
                     seed=args.seed,
                     tau=args.tau,
                     gamma=args.gamma,
                     lr=args.lr,
                     lr_actor=args.lr_actor,
                     auto_entropy_tuning=args.auto_entropy_tuning,
                     backup_entropy=args.backup_entropy,
                     target_entropy=args.target_entropy,
                     with_lagrange=args.with_lagrange,
                     lagrange_thresh=args.lagrange_thresh)
    print(f"\nThe actor architecture is:\n{jax.tree_map(lambda x: x.shape, agent.actor_state.params)}")
    print(f"\nThe critic architecture is:\n{jax.tree_map(lambda x: x.shape, agent.critic_state.params)}")

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

        # save some evaluate time
        if ((t+1) >= int(9.5e5) and (t + 1) % args.eval_freq == 0) or ((t+1) <= int(9.5e5) and (t + 1) % (2*args.eval_freq) == 0):
            eval_reward = eval_policy(agent, args.env, args.seed)
            log_info.update({
                "step": t+1,
                "reward": eval_reward,
                "time": (time.time() - start_time) / 60
            })
            logs.append(log_info)
            print(
                f"\n# Step {t+1}: {eval_reward:.2f}, critic_loss: {log_info['critic_loss']:.2f}, "
                f"actor_loss: {log_info['actor_loss']:.2f}, alpha_loss: {log_info['alpha_loss']:.2f}, "
                f"cql1_loss: {log_info['cql1_loss']:.2f}, cql2_loss: {log_info['cql2_loss']:.2f}, "
                f"q1: {log_info['q1']:.2f}, q2: {log_info['q2']:.2f}, "
                f"cql_q1: {log_info['cql_q1']:.2f}, cql_q2: {log_info['cql_q2']:.2f}, "
                f"cql_next_q1: {log_info['cql_next_q1']:.2f}, cql_next_q2: {log_info['cql_next_q2']:.2f}, "
                f"random_q1: {log_info['random_q1']:.2f}, random_q2: {log_info['random_q2']:.2f}, "
                f"alpha: {log_info['alpha']:.2f}, logp: {log_info['logp']:.2f}, "
                f"logp_next_action: {log_info['logp_next_action']:.2f}"
            )


    # Save logs
    log_name = f"s{args.seed}"
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/{args.env}", exist_ok=True)
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{args.log_dir}/{args.env}/{log_name}.csv")
    with open(f"{args.log_dir}/{args.env}/{log_name}.json", "w") as f:
        json.dump(vars(args), f)
    # agent.save(f"{args.model_dir}/{args.env}/{args.seed}")


if __name__ == "__main__":
    args = get_args()
    print(f"\nArguments:\n{vars(args)}")
    main(args)
