import d4rl
import gym
import jax
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import numpy as np
import pandas as pd

from tqdm import trange
from models.combo import COMBOAgent
from models.td3bc import TD3BCAgent
from models.td3 import TD3Agent
from models.cql import CQLAgent
from utils import ReplayBuffer, load_data, get_s_embeddings, get_sa_embeddings


AGENTS = {'cql': CQLAgent, 'td3bc': TD3BCAgent, 'td3': TD3Agent, 'combo': COMBOAgent}
baseline_agent_df = pd.read_csv('config/baseline_agent.csv', index_col=0)
baseline_agent_df = baseline_agent_df.set_index(['env_name', 'algo', 'seed'])


# Eval agent for 10 runs
def eval_policy(agent, env_name, mu, std, seed=0, eval_episodes=10):
    eval_env = gym.make(env_name)
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


# Load test dataset collected by an online TD3 agent
def load_td3_test_data(env_name, L=50):
    test_data = np.load(f"saved_buffers/{env_name.split('-')[0]}-v2/L{L}K.npz")
    test_obs = test_data["observations"]
    test_act = test_data["actions"]
    return test_obs, test_act


def run_repr_rank_exp():
    res = []
    for env_name in ['halfcheetah-medium-replay-v2', 'hopper-medium-v2', 'walker2d-medium-expert-v2']:
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # normalize state stats for TD3BC
        replay_buffer = ReplayBuffer(obs_dim, act_dim)
        replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
        mu, std = replay_buffer.normalize_states()

        # load probing data
        observations, actions, _, _ = load_data(env_name.split('-')[0]+'-v2', 'L100K')
        randn_idx = np.random.permutation(np.arange(len(observations)))

        # probing experiment w.r.t. checkpoints
        for algo in ['td3bc', 'cql', 'combo']:
            agent = AGENTS[algo](obs_dim=obs_dim, act_dim=act_dim)
            for seed in range(10):
                step = baseline_agent_df.loc[(env_name, algo, seed), 'step']

                # Check score before/after training
                agent.load(f"saved_agents/{algo}/{env_name}/s{seed}_{step}")

                for tmp_L in [10000, 30000, 50000]:
                    tmp_idx = randn_idx[:tmp_L]
                    s_embeddings = get_s_embeddings(agent, observations[tmp_idx], mu, std)
                    sa_embeddings = get_sa_embeddings(agent, observations[tmp_idx], actions[tmp_idx], mu, std)
                    mean_s_embedding = s_embeddings.mean(axis=0)
                    mean_sa_embedding = sa_embeddings.mean(axis=0)
                    num_s_zero_col = sum(abs(mean_s_embedding) < 1e-6)
                    num_sa_zero_col = sum(abs(mean_sa_embedding) < 1e-6)
                    normalized_s_embeddings = s_embeddings - mean_s_embedding
                    normalized_sa_embeddings = sa_embeddings - mean_sa_embedding

                    # approximate covariance matrix with samples
                    s_cov = np.zeros(shape=(256, 256))
                    for tmp_idx in range(int(np.ceil(tmp_L / 5000))):
                        tmp_embeddings = normalized_s_embeddings[tmp_idx*5000: (tmp_idx+1)*5000]
                        tmp_s_cov = np.matmul(tmp_embeddings.reshape(-1, 256, 1), tmp_embeddings.reshape(-1, 1, 256)).sum(0)
                        s_cov += tmp_s_cov
                    s_cov /= tmp_L

                    sa_cov = np.zeros(shape=(256, 256))
                    for tmp_idx in range(int(np.ceil(tmp_L / 5000))):
                        tmp_embeddings = normalized_sa_embeddings[tmp_idx*5000: (tmp_idx+1)*5000]
                        tmp_sa_cov = np.matmul(tmp_embeddings.reshape(-1, 256, 1), tmp_embeddings.reshape(-1, 1, 256)).sum(0)
                        sa_cov += tmp_sa_cov
                    sa_cov /= tmp_L

                    _, sigma_s, _ = np.linalg.svd(s_cov, full_matrices=False)
                    _, sigma_sa, _ = np.linalg.svd(sa_cov, full_matrices=False)

                    res.append((env_name, algo, seed, tmp_L, num_s_zero_col, 'actor_embedding', *sigma_s))
                    res.append((env_name, algo, seed, tmp_L, num_sa_zero_col, 'critic_embedding', *sigma_sa))
    
    res_df = pd.DataFrame(res, columns=['env_name', 'algo', 'seed', 'size', 'zero_col_num', 'embedding']+[f'sigma{i}' for i in range(1, 257)])
    res_df.to_csv('probe_exp_res/repr_rank_res.csv')
