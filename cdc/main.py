import gym
import d4rl
import jax
import os
import time
import yaml
import logging
import numpy as np
import pandas as pd
from tqdm import trange
from models import CDCAgent
from utils import ReplayBuffer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"


def eval_policy(agent: CDCAgent,
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
            eval_rng, action = agent.select_action(agent.actor_state.params, eval_rng, np.array(obs), True)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr", default=7e-4, type=float)
    parser.add_argument("--lr_actor", default=3e-4, type=float)
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--eval_freq", default=int(5e3), type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--nu", default=0.75, type=float)
    parser.add_argument("--lmbda", default=1.0, type=float)
    parser.add_argument("--eta", default=1.0, type=float)
    parser.add_argument("--num_samples", default=15, type=int)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--config_dir", default="./configs", type=str)
    parser.add_argument("--model_dir", default="./saved_models", type=str)
    args = parser.parse_args()
    return args


def main(args):
    # Experiment name
    exp_name = f's{args.seed}'
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
    logger.info(f"Arguments:\n{vars(args)}")

    # Env parameters
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # random seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)

    # CDC agent
    agent = CDCAgent(obs_dim=obs_dim,
                     act_dim=act_dim,
                     seed=args.seed,
                     nu=args.nu,
                     eta=args.eta,
                     tau=args.tau,
                     gamma=args.gamma,
                     lmbda=args.lmbda,
                     num_samples=args.num_samples,
                     lr=args.lr,
                     lr_actor=args.lr_actor)

    logger.info(f"\nThe actor architecture is:\n{jax.tree_map(lambda x: x.shape, agent.actor_state.params)}")
    logger.info(f"\nThe critic architecture is:\n{jax.tree_map(lambda x: x.shape, agent.critic_state.params)}\n")

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    fix_obs = np.random.normal(size=(128, obs_dim))
    fix_act = np.random.normal(size=(128, act_dim))

    # Evaluate the untrained policy
    logs = [{"step": 0, "reward": eval_policy(agent, args.env, args.seed)}]

    # Initialize training stats
    start_time = time.time()

    # Train agent and evaluate policy
    for t in trange(args.max_timesteps):
        log_info = agent.update(replay_buffer, args.batch_size)

        # save some evaluate time
        if ((t + 1) >= int(9.5e5) and
            (t + 1) % args.eval_freq == 0) or ((t + 1) <= int(9.5e5) and
                                               (t + 1) %
                                               (2 * args.eval_freq) == 0):
            eval_reward = eval_policy(agent, args.env, args.seed)
            log_info.update({
                "step": t + 1,
                "reward": eval_reward,
                "time": (time.time() - start_time) / 60
            })
            logs.append(log_info)
            fix_q = agent.critic.apply(
                {"params": agent.critic_state.params}, fix_obs, fix_act)
            _, fix_a = agent.select_action(agent.actor_state.params,
                                           jax.random.PRNGKey(0),
                                           fix_obs, True)
            logger.info(
                f"\n# Step {t+1}: eval_reward = {eval_reward:.2f}, time: {log_info['time']:.2f}\n"
                f"\tcritic_loss: {log_info['critic_loss']:.2f}, penalty_loss: {log_info['penalty_loss']:.2f}\n"
                f"\tactor_loss: {log_info['actor_loss']:.2f}, mle_prob: {log_info['mle_prob']:.2f}\n"
                f"\tconcat_q_avg: {log_info['concat_q_avg']:.2f}, concat_q_min: {log_info['concat_q_min']:.2f}, concat_q_max: {log_info['concat_q_max']:.2f}\n"
                f"\ttarget_q: {log_info['target_q']:.2f}, fix_q: {fix_q.squeeze().mean().item():.2f}, fix_a: {abs(fix_a).sum().item():.2f}\n\n"
            )

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{args.log_dir}/{args.env}/{exp_name}.csv")


if __name__ == "__main__":
    args = get_args()
    print(f"\nArguments:\n{vars(args)}")
    with open(f"{args.config_dir}/cdc.yaml", "r") as stream:
        configs = yaml.safe_load(stream)
    args.eta = configs[args.env]["eta"]
    args.lmbda = configs[args.env]["lmbda"]
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/{args.env}", exist_ok=True)
    main(args)
