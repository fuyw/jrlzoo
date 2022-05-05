import d4rl
import gym
import jax
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import numpy as np
import pandas as pd

from tqdm import tqdm
from models.combo import COMBOAgent
from models.td3bc import TD3BCAgent
from models.td3 import TD3Agent
from models.cql import CQLAgent
from utils import ReplayBuffer


AGENTS = {'td3bc': TD3BCAgent, 'combo': COMBOAgent, 'cql': CQLAgent, 'td3': TD3Agent}


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


# 0. Check COMBO agent's weight
def check_combo_agent_weight():
    env = gym.make('walker2d-medium-expert-v2')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # initialize the agent
    agent1 = AGENTS['combo'](obs_dim=obs_dim, act_dim=act_dim)
    agent1.load(f"saved_agents/combo/walker2d-medium-expert-v2/s9_200")

    agent2 = AGENTS['combo'](obs_dim=obs_dim, act_dim=act_dim)
    agent2.load(f"saved_agents/combo/walker2d-medium-expert-v2/s1_200")

    agent1.actor_state.params['fc1']['kernel']
    agent2.actor_state.params['fc1']['kernel']

    agent1.actor_state.params['fc2']['bias']
    agent2.actor_state.params['fc2']['bias']


# 1. Evaluate `algo` checkpoints ==> 50 checkpoints/algo
def eval_agent(algo, env_name="hopper-medium-v2"):
    res = []
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # initialize the agent
    agent = AGENTS[algo](obs_dim=obs_dim, act_dim=act_dim)

    # normalize state stats for TD3BC
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    mu, std = replay_buffer.normalize_states()

    # evaluate saved checkpoint, 5 checkpoints per random seed
    for seed, step in tqdm([(i,j) for i in range(10) for j in range(196, 201)],
                           desc='[Evaluate checkpoints]'):
        agent.load(f"saved_agents/{algo}/{env_name}/s{seed}_{step}")
        trained_score = eval_policy(agent, env_name, mu, std, seed)
        res.append((env_name, seed, step, trained_score))

    res_df = pd.DataFrame(res, columns=['env', 'seed', 'step', 'reward'])
    res_df.to_csv(f'eval_agent_res/{algo}/{env_name}_agent.csv')


# 2. Select the optimal online TD3 agent checkpoint
def select_optimal_td3_agent():
    """
    with open('config/optimal_td3_agent.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    """
    res = {}
    for env_name in ['halfcheetah-medium-v2',
                     'hopper-medium-v2',
                     'walker2d-medium-v2']:
        res_df = pd.read_csv(f'eval_agent_res/td3/{env_name}_agent.csv', index_col=0)
        optimal_idx = res_df['reward'].argmax()
        res[env_name] = dict(
            seed=str(res_df.iloc[optimal_idx]['seed']),
            step=str(res_df.iloc[optimal_idx]['step']))

    with open('config/optimal_td3_agent.yaml', 'w') as f:
        yaml.dump(res, f, default_flow_style=False)


# 3. Select the baseline agent checkpoint
def select_baseline_agent():
    res = []
    for env_name in ['halfcheetah-medium-replay-v2',
                     'hopper-medium-v2',
                     'walker2d-medium-expert-v2']:
        for algo in ['combo', 'cql', 'td3bc']:
            df = pd.read_csv(f'eval_agent_res/{algo}/{env_name}_agent.csv', index_col=0)
            median_reward = df['reward'].median()
            df['delta'] = abs(df['reward'] - median_reward)
            for seed in range(10):
                seed_df = df.query(f'seed == {seed}')
                min_idx = seed_df['delta'].argmin()
                step = seed_df.iloc[min_idx]['step']
                res.append((env_name, algo, seed, step))
    res_df = pd.DataFrame(res, columns=['env_name', 'algo', 'seed', 'step'])
    res_df.to_csv('config/baseline_agent.csv')



if __name__ == "__main__":
    os.makedirs('config', exist_ok=True)
    os.makedirs('eval_agent_res', exist_ok=True)
    os.makedirs('eval_agent_res/cql', exist_ok=True)

    # for algo in ['td3bc', 'cql', 'combo']:
    #     os.makedirs(f'eval_agent_res/{algo}', exist_ok=True)
    #     for env_name in ['halfcheetah-medium-replay-v2',
    #                      'hopper-medium-v2',
    #                      'walker2d-medium-expert-v2']:
    #         eval_agent(algo, env_name)

    for env_name in ['hopper-medium-v2',
                     'hopper-medium-replay-v2',
                     'hopper-medium-expert-v2',
                     'halfcheetah-medium-replay-v2',
                     'halfcheetah-medium-expert-v2', 
                     'walker2d-medium-expert-v2'
                    #  'halfcheetah-medium-v2', 
                     ]:
        eval_agent('cql', env_name)
