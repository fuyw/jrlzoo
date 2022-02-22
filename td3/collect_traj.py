import os
import copy
import gym
import numpy as np
from tqdm import tqdm
from models import TD3
from utils import ReplayBuffer
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"


step_dict = {
    'Hopper-v2': [10, 25, 40, 100],
    'Walker2d-v2': [10, 30, 50, 100],
    'HalfCheetah-v2': [10, 30, 50, 100]
}


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
    parser.add_argument("--env", default="Hopper-v2")
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

    # Load saved agents
    agent_dict = {}
    steps = step_dict[args.env]
    agent = TD3(obs_dim=obs_dim, act_dim=act_dim, max_action=max_action)
    agent_dict[0] = copy.deepcopy(agent)
    for step in steps:
        agent.load(f"saved_models/{args.env}/step{step}_seed0")
        agent_dict[step] = copy.deepcopy(agent)

    # Eval agents
    for step in agent_dict.keys():
        eval_reward = eval_policy(agent_dict[step], args.env, 0)
        print(f"[Step - {step}][{args.env}] Eval reward = {eval_reward:.2f}")

    # Collect trajectories
    replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(1e6))
    traj_len = 100000
    for step in tqdm(agent_dict.keys(), desc='[Collect trajectories]'):
        agent = agent_dict[step]
        t = 0
        flag = True
        while flag:
            obs, done = env.reset(), False
            while (not done) and flag:
                t += 1
                if np.random.random() < 0.3:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(agent.actor_state.params, np.array(obs))
                next_obs, reward, done, _ = env.step(action)
                replay_buffer.add(obs, action, next_obs, reward, done)
                obs = next_obs
                if t == traj_len:
                    flag = False

    # Save buffer
    os.makedirs(f'saved_buffers', exist_ok=True)
    os.makedirs(f'saved_buffers/{args.env}', exist_ok=True)
    replay_buffer.save(f'saved_buffers/{args.env}/5agents')


if __name__ == "__main__":
    args = get_args()
    for env in ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2']:
        args.env = env
        main(args)
