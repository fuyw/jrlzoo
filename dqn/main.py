import bsuite
from bsuite import sweep
from bsuite.utils import gym_wrapper
import numpy as np

from models import DQN
from utils import ReplayBuffer, evaluate_catch


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="catch/0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_episodes", default=10000, type=int)
    parser.add_argument("--min_replay_size", default=100, type=int)
    parser.add_argument("--eval_episodes", default=100, type=int)
    parser.add_argument("--epsilon", default=0.05, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--batch_size", default=32, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--target_update_period", default=4, type=int)
    args = parser.parse_args()
    return args


def main(args):
    sweep.SETTINGS["catch/0"]["seed"] = args.seed
    bsuite_env = bsuite.load_from_id(bsuite_id=args.env)
    env = gym_wrapper.GymFromDMEnv(bsuite_env)
    np.random.seed(args.seed)

    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.n
    print(f"Run DQN on Env {args.env}: obs_dim = {obs_dim}, act_dim = {act_dim}")

    agent = DQN(obs_dim, act_dim, args.learning_rate, args.gamma,
                args.seed, args.target_update_period)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim)

    for episode in range(args.num_episodes):
        obs, done = env.reset(), False
        while not done:
            if np.random.random() < args.epsilon:
                action = np.random.randint(0, 3)
            else:
                action = agent.select_action(agent.state.params, obs.flatten())
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.add(obs, action, next_obs, reward, float(done))
            obs = next_obs
            if replay_buffer.size >= args.min_replay_size:
                log_info = agent.train(replay_buffer, args.batch_size)

        if (episode + 1) % args.eval_episodes == 0:
            eval_reward = evaluate_catch(agent)
            if episode >= args.min_replay_size:
                print(f"# Episode {episode + 1}: reward = {eval_reward:.2f}, "
                    f"q_loss = {log_info['q_loss']:.2f}, q = {log_info['q']:.2f}")


if __name__ == "__main__":
    args = get_args()
    print(f"\nArguments:\n{vars(args)}")
    main(args)
