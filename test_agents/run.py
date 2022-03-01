import d4rl
import gym
import jax
import os
import numpy as np
import pandas as pd
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

from tqdm import trange
from models.combo import COMBOAgent
from models.td3bc import TD3BCAgent
from models.cql import CQLAgent
from probe import ProbeTrainer
from utils import *


def eval_policy(agent, env_name, mu, std, seed=0, eval_episodes=10):
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
    parser.add_argument("--target", default="action")
    args = parser.parse_args()
    return args


args = get_args()
args.env_name = "hopper-medium-v2"
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
def probe_next_observations(embeddings, next_observations):
    trainer = ProbeTrainer(256, obs_dim)
    kf_losses = trainer.train(embeddings, next_observations)
    return kf_losses

# predict a* based on 𝜙(s, s')
def probe_actions(embeddings, actions):
    trainer = ProbeTrainer(512, act_dim)
    kf_losses = trainer.train(embeddings, actions)
    return kf_losses


def probe_agents(args):
    for Agent, algo in [
            # (COMBOAgent, 'combo'),
            (TD3BCAgent, 'td3bc'),
            # (CQLAgent, 'cql')
    ]:
        seed = AGENT_DICTS[args.env_name][algo]
        agent = Agent(obs_dim=obs_dim, act_dim=act_dim)

        # Check score before/after training
        untrained_score = eval_policy(agent, args.env_name, mu, std)
        print(f"Score before training: {untrained_score:.2f}")  # 0.8,   1.7
        agent.load(f"saved_agents/{algo}/{args.env_name}/s{seed}")
        trained_score = eval_policy(agent, args.env_name, mu, std)
        print(f"Score after training: {trained_score:.2f}")     # 67.25, 94.41

        # load probing data
        observations, actions, rewards, next_observations = load_data(args)
        if args.target == 'action':
            embeddings = get_ss_embeddings(args, agent, observations, next_observations, mu, std)  # (N, 512)
        else:
            embeddings = get_sa_embeddings(args, agent, observations, actions, mu, std)  # (N, 256)

        if args.target == 'reward':
            kf_losses = probe_rewards(embeddings, rewards)
        elif args.target == 'next_obs':
            kf_losses = probe_next_observations(embeddings, next_observations)
        elif args.target == 'action':
            kf_losses = probe_actions(embeddings, actions)

        df = pd.DataFrame(kf_losses, columns=['cv_loss'])
        df.to_csv(f'res/{args.env_name}/{algo}_probe_{args.target}.csv')


def random_baseline(args):
    observations, actions, rewards, next_observations = load_data(args)
    for target in ['reward', 'next_obs', 'action']:
        if target == 'action':
            embeddings = np.random.normal(size=(len(observations), 512))
        else:
            embeddings = np.random.normal(size=(len(observations), 256))

        if target == 'reward':
            kf_losses = probe_rewards(embeddings, rewards)
        elif target == 'next_obs':
            kf_losses = probe_next_observations(embeddings, next_observations)
        elif target == 'action':
            kf_losses = probe_actions(embeddings, actions)

        df = pd.DataFrame(kf_losses, columns=['cv_loss'])
        df.to_csv(f'res/{args.env_name}/random_probe_{target}.csv')


args.target = 'action'
os.makedirs(f'res/{args.env_name}', exist_ok=True)
probe_agents(args)
