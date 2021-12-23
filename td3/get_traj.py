import gym
import numpy as np
from models import TD3


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
    args = parser.parse_args()
    return args


def main(args):
    # Env parameters
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(obs_dim=obs_dim, act_dim=act_dim, max_action=max_action)
    eval_reward1 = eval_policy(agent, args.env, 0)
    print(f"Eval reward before load parameters: {eval_reward1}")
    agent.load(f"saved_models/Walker2d-v2/8")
    eval_reward2 = eval_policy(agent, args.env, 0)
    print(f"Eval reward before load parameters: {eval_reward2}")


if __name__ == "__main__":
    args = get_args()
    main(args)
