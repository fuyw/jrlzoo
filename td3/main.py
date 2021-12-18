import argparse
import gym
import os
import time
import numpy as np
import pandas as pd

from models import TD3
from utils import ReplayBuffer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"


def eval_policy(agent: TD3,
                env_name: str,
                seed: int,
                eval_episodes: int = 10) -> float:
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        t = 0
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            action = agent.select_action(agent.actor_state.params, np.array(obs))
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--start_timesteps", default=int(25e3), type=int)
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--eval_freq", default=int(5e3), type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--model_dir", default="./saved_models", type=str)
    args = parser.parse_args()
    return args


def main(args):
    # Env parameters
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # andom seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)

    # TD3 agent
    agent = TD3(obs_dim=obs_dim,
                act_dim=act_dim,
                max_action=max_action,
                tau=args.tau,
                gamma=args.gamma,
                noise_clip=args.noise_clip,
                policy_noise=args.policy_noise,
                policy_freq=args.policy_freq,
                learning_rate=args.learning_rate,
                seed=args.seed)

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)

    # Evaluate the untrained policy
    logs = [{"step": 0, "reward": eval_policy(agent, args.env, args.seed)}]

    # Initialize training stats
    obs, done = env.reset(), False
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    start_time = time.time()

    # Train agent and evaluate policy
    for t in range(args.max_timesteps):
        episode_timesteps += 1
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                agent.select_action(agent.actor_state.params, np.array(obs)) +
                np.random.normal(0, max_action * args.expl_noise,
                                 size=act_dim)).clip(-max_action, max_action)

        next_obs, reward, done, _ = env.step(action)
        done_bool = float(
            done) if episode_timesteps < env._max_episode_steps else 0
        replay_buffer.add(obs, action, next_obs, reward, done_bool)

        obs = next_obs
        episode_reward += reward

        if t >= args.start_timesteps:
            log_info = agent.train(replay_buffer, args.batch_size)

        if done:
            obs, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % args.eval_freq == 0:
            eval_reward = eval_policy(agent, args.env, args.seed)
            if t >= args.start_timesteps:
                log_info.update({
                    "step": t+1,
                    "reward": eval_reward,
                    "time": (time.time() - start_time) / 60
                })
                logs.append(log_info)
                print(
                    f"# Step {t+1}: {eval_reward:.2f}, critic_loss: {log_info['critic_loss']:.3f}, "
                    f"q1: {log_info['q1']:.3f}, q2: {log_info['q2']:.3f}")
            else:
                logs.append({"step": t+1, "reward": eval_reward})
                print(f"# Step {t+1}: {eval_reward:.2f}")

    # Save logs
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/{args.env}", exist_ok=True)
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{args.log_dir}/{args.env}/{args.seed}.csv")
    agent.save(f"{args.model_dir}/{args.env}/{args.seed}")


if __name__ == "__main__":
    args = get_args()
    print(f"\nArguments:\n{vars(args)}")
    env = gym.make(args.env)
    main(args)
