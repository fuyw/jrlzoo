import d4rl
import gym
import os
import time
import logging
import numpy as np
import pandas as pd
from tqdm import trange

from models import TD3BCM_Agent
from utils import ReplayBuffer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"


def eval_policy(agent: TD3BCM_Agent,
                env_name: str,
                seed: int,
                mean: np.ndarray,
                std: np.ndarray,
                eval_episodes: int = 10) -> float:
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        t = 0
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            obs = (np.array(obs).reshape(1, -1) - mean) / std
            action = agent.select_action(agent.actor_state.params, obs.squeeze())
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="td3bc")
    parser.add_argument("--env_name", default="halfcheetah-medium-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--eval_freq", default=int(5e3), type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--alpha", default=2.5, type=float)
    parser.add_argument("--normalize", default=True, action="store_false")
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--model_dir", default="./saved_models", type=str)
    parser.add_argument("--dynamics_model_dir", default="./saved_dynamics_models", type=str)
    parser.add_argument("--horizon", default=3, type=int)
    args = parser.parse_args()
    return args


def main(args):
    exp_name = f'td3bcm_s{args.seed}_h{args.horizon}'
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
    max_action = float(env.action_space.high[0])

    # random seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)

    # Replay D4RL buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
        print(f"Get normalized states.")
    else:
        mean, std = 0, 1
    model_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(args.horizon*1e5))
    agent = TD3BCM_Agent(env_name=args.env_name,
                         obs_dim=obs_dim,
                         act_dim=act_dim,
                         max_action=max_action,
                         tau=args.tau,
                         gamma=args.gamma,
                         noise_clip=args.noise_clip,
                         policy_noise=args.policy_noise,
                         policy_freq=args.policy_freq,
                         learning_rate=args.learning_rate,
                         alpha=args.alpha,
                         seed=args.seed,
                         horizon=args.horizon,
                         mu=mean,
                         std=std)

    # Load the trained dynamics model
    agent.model.load(f'{args.dynamics_model_dir}/{args.env_name}')

    # Evaluate the untrained policy
    logs = [{"step": 0, "reward": eval_policy(agent, args.env_name, args.seed, mean, std)}]

    # Initialize training stats
    start_time = time.time()

    # Train agent and evaluate policy
    for t in trange(args.max_timesteps):
        log_info = agent.train(replay_buffer, model_buffer)

        if (t + 1) % args.eval_freq == 0:
            eval_reward = eval_policy(agent, args.env_name, args.seed, mean, std)
            log_info.update({
                "step": t+1,
                "reward": eval_reward,
                "time": (time.time() - start_time) / 60
            })
            logs.append(log_info)
            logger.info(
                f"\n[# Step {t+1}] eval_reward: {eval_reward:.2f}\n"
                f"\tcritic_loss: {log_info['critic_loss']:.3f}, actor_loss: {log_info['actor_loss'].item():.3f}, "
                f"bc_loss: {log_info['bc_loss'].item():.3f}\n"
                f"\tq1: {log_info['q1']:.3f}, q2: {log_info['q2']:.3f}, target_q: {log_info['target_q']:.3f}\n"
                f"\treal_batch_rewards: {log_info['real_batch_rewards']:.2f}, "
                f"real_batch_actions: {log_info['real_batch_actions']:.2f}, "
                f"real_batch_discounts: {log_info['real_batch_discounts']:.2f}\n"
                f"\tmodel_batch_rewards: {log_info['model_batch_rewards']:.2f}, "
                f"model_batch_actions: {log_info['model_batch_actions']:.2f}, "
                f"model_batch_discounts: {log_info['model_batch_discounts']:.2f}\n"
                f"\tmodel_buffer_size: {log_info['model_buffer_size']:.0f}, "
                f"model_buffer_ptr: {log_info['model_buffer_ptr']:.0f}\n"
            )

        if ((t + 1) >= int(9.8e5) and (t + 1) % args.eval_freq == 0) :
            agent.save(f"{args.model_dir}/{args.env_name}/s{args.seed}_{(t + 1) // args.eval_freq}")

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{args.log_dir}/{args.env_name}/s{args.seed}.csv")



if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env_name}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/{args.env_name}", exist_ok=True)
    main(args)
