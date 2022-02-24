import d4rl
import gym
import jax
import os
import numpy as np
import pandas as pd
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

from tqdm import trange
from models.combo import COMBOAgent
from models.td3bc import TD3BCAgent
from models.cql import CQLAgent
from mlp import ProbeTrainer
from utils import ReplayBuffer, load_data, get_embeddings


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
            if isinstance(agent, COMBOAgent) or isinstance(agent, CQLAgent):
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
args.env_name = "walker2d-medium-v2"
env = gym.make(args.env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# normalize state stats for TD3BC
replay_buffer = ReplayBuffer(obs_dim, act_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
mu, std = replay_buffer.normalize_states()


# predict `r` based on 𝜙(s, a)
def probe_rewards(embeddings, rewards):
    trainer = ProbeTrainer(256, 1)
    kf_losses = trainer.train(embeddings, rewards.reshape(-1, 1))
    return kf_losses

# predict s' based on 𝜙(s, a)
def probe_next_observation(embeddings, next_observations):
     

for Agent, agent_name, seed in [
        # (TD3BCAgent, 'td3bc', 0),
        # (COMBOAgent, 'combo', 3)
        (CQLAgent, 'cql', 42)
]:
    agent = Agent(obs_dim=obs_dim, act_dim=act_dim)
    untrained_score = eval_policy(agent, args.env_name)
    print(f"Score before training: {untrained_score:.2f}")  # 0.8,   1.7
    agent.load(f"saved_agents/{agent_name}/{args.env_name}/s{seed}")
    trained_score = eval_policy(agent, args.env_name)
    print(f"Score after training: {trained_score:.2f}")     # 67.25, 94.41

    observations, actions, rewards, next_observations = load_data(args)
    embeddings = get_embeddings(args, agent, observations, actions)
    reward_kf_losses = probe_rewards(embeddings, rewards)
    df = pd.DataFrame(reward_kf_losses, columns=['reward'])
    df.to_csv(f'res/{agent_name}_probe_reward.csv')

    # using random embeddings
    random_embeddings = np.random.normal(size=embeddings.shape)
    reward_kf_losses = probe_rewards(random_embeddings, rewards)
    df = pd.DataFrame(reward_kf_losses, columns=['reward'])
    df.to_csv(f'res/random_probe_reward.csv')
