import d4rl
import gym
import jax
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

from models.combo import COMBOAgent
from models.td3bc import TD3BCAgent
from utils import ReplayBuffer


env_name = 'hopper-medium-v2'


def eval_policy(agent, env_name, seed=0, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    eval_rng = jax.random.PRNGKey(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        t = 0
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            if isinstance(agent, COMBOAgent):
                eval_rng, action = agent.select_action(agent.actor_state.params, eval_rng, obs, True)
            elif isinstance(agent, TD3BCAgent):
                obs = (obs - mu)/std
                action = agent.select_action(agent.actor_state.params, obs.squeeze())
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="hopper-medium-v2")
    args = parser.parse_args()
    return args


args = get_args()
env = gym.make(args.env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# normalize state stats for TD3BC
replay_buffer = ReplayBuffer(obs_dim, act_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
mu, std = replay_buffer.normalize_states()


agent = TD3BCAgent(obs_dim=obs_dim, act_dim=act_dim)
# agent = COMBOAgent(obs_dim=obs_dim, act_dim=act_dim)
untrained_score = eval_policy(agent, args.env_name)
agent.load(f"saved_agents/td3bc/{args.env_name}/s0")
# agent.load("saved_agents/combo/s3")
trained_score = eval_policy(agent, args.env_name)
print(f"Score before training: {untrained_score:.2f}")  # 1.7
trained_score = eval_policy(agent, args.env_name)
print(f"Score after training: {trained_score:.2f}")     # 94.41


def load_data():
    data = np.load(f"saved_buffers/{args.env_name.split('-')[0]}/5agents.npz")
    observations = data["observations"]
    actions = data["actions"]
    next_observations = data["next_observations"]
    rewards = data["rewards"]
    discounts = data["discounts"]
