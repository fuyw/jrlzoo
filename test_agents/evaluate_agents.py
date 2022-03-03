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
from models.td3 import TD3Agent
from models.cql import CQLAgent
from probe import ProbeTrainer
from utils import ReplayBuffer, load_data, AGENT_DICTS


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
            elif isinstance(agent, TD3Agent):
                action = agent.select_action(agent.actor_state.params, obs)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score


AGENTS = {'td3bc': TD3BCAgent, 'combo': COMBOAgent, 'cql': CQLAgent, 'td3': TD3Agent}


def check_agent(algo, env_name="hopper-medium-v2"):
    res = []
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # normalize state stats for TD3BC
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    mu, std = replay_buffer.normalize_states()

    for seed in trange(10):
        agent = AGENTS[algo](obs_dim=obs_dim, act_dim=act_dim)
        for step in range(196, 201):
            agent.load(f"saved_agents/{algo}/{env_name}/s{seed}_{step}")
            trained_score = eval_policy(agent, env_name, mu, std)
            res.append((env_name, seed, step, trained_score))

    res_df = pd.DataFrame(res, columns=['env', 'seed', 'step', 'reward'])
    res_df.to_csv(f'eval_agent_res/{algo}/{env_name}_agent.csv')



def select_seed_step(algo, env_name):
    reward_df = pd.read_csv('eval_agent_res/')
    pass


def select_optimal_td3_agent(env_name):
    res_df = pd.read_csv(f'eval_agent_res/td3/{env_name}_agent.csv', index_col=0)
    optimal_idx = res_df['reward'].argmax()
    return res_df.iloc[optimal_idx][['seed', 'step']].values


if __name__ == "__main__":
    os.makedirs('eval_agent_res', exist_ok=True)

    algo = 'td3bc'
    os.makedirs(f'eval_agent_res/{algo}', exist_ok=True)

    check_agent(algo, 'hopper-medium-v2')
