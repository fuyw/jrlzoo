import d4rl
import gym
import jax
import os
import numpy as np
import pandas as pd
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"

from models.combo import COMBOAgent
from models.td3bc import TD3BCAgent
from models.cql import CQLAgent
from models.td3 import TD3Agent
from probe_trainer import ProbeTrainer
from utils import *


# Evaluate agent for 10 runs
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
            else:
                raise Exception('Unknown agent')
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score


# Load optimal TD3 agents
def get_optimal_td3_agents():
    optimal_td3_df = pd.read_csv('config/optimal_online_td3.csv',
                                 index_col=0).set_index('env_name')
    optimal_td3_agents = {}
    for env_name in optimal_td3_df.index:
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        seed, step = optimal_td3_df.loc[env_name, ['seed', 'step']]
        agent = AGENTS['td3'](obs_dim=obs_dim, act_dim=act_dim)

        # Check score before/after training
        agent.load(f"saved_agents/td3/{env_name}/s{seed}_{step}")
        eval_reward = eval_policy(agent, env_name, mu=None, std=None, seed=0, eval_episodes=10)
        print(f'Optimal TD3 agent score for {env_name} is {eval_reward:.3f}')
        optimal_td3_agents[env_name.split('-')[0]] = agent
    return optimal_td3_agents


# Get experiment args
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="hopper-medium-v2")
    parser.add_argument("--target", default="action")
    args = parser.parse_args()
    return args


AGENTS = {'cql': CQLAgent, 'td3bc': TD3BCAgent, 'td3': TD3Agent, 'combo': COMBOAgent}
baseline_agent_df = pd.read_csv('config/baseline_agent.csv', index_col=0)
baseline_agent_df = baseline_agent_df.set_index(['env_name', 'algo', 'seed'])


# predict `r` based on 𝜙(s, a)
def probe_rewards(embeddings, rewards, epochs=100):
    trainer = ProbeTrainer(embeddings.shape[1], 1, epochs=epochs)
    kf_losses = trainer.train(embeddings, rewards.reshape(-1, 1))
    return kf_losses


# predict `V(s)` or `Q(s, a)` based on 𝜙(s) and 𝜙(s, a)
def probe_values(embeddings, values, epochs=300):
    trainer = ProbeTrainer(embeddings.shape[1], 1, epochs=epochs)
    kf_losses = trainer.train(embeddings, values.reshape(-1, 1))
    return kf_losses


# predict s' based on 𝜙(s, a)
def probe_next_observations(embeddings, next_observations, epochs=100):
    trainer = ProbeTrainer(embeddings.shape[1], next_observations.shape[1], epochs=epochs)
    kf_losses = trainer.train(embeddings, next_observations)
    return kf_losses

# predict a* based on 𝜙(s, s')
def probe_actions(embeddings, actions, epochs=100):
    trainer = ProbeTrainer(embeddings.shape[1], actions.shape[1], epochs=epochs)
    kf_losses = trainer.train(embeddings, actions)
    return kf_losses


# random embedding exp
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


# Probing experiments for `dynamics` information
def run_dynamics_probing_exp(args):
    res = []
    for env_name in ['halfcheetah-medium-replay-v2', 'hopper-medium-v2', 
                     'walker2d-medium-expert-v2']:
        args.env_name = env_name
        env = gym.make(args.env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # normalize state stats for TD3BC
        replay_buffer = ReplayBuffer(obs_dim, act_dim)
        replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
        mu, std = replay_buffer.normalize_states()

        # load probing data
        observations, actions, rewards, next_observations = load_data(
            env_name.split('-')[0]+'-v2', 'L100K')
 
        for algo in ['td3bc', 'cql', 'combo']:
            agent = AGENTS[algo](obs_dim=obs_dim, act_dim=act_dim)
            for seed in range(10):
                step = baseline_agent_df.loc[(env_name, algo, seed), 'step']

                # Check score before/after training
                agent.load(f"saved_agents/{algo}/{args.env_name}/s{seed}_{step}")

                if args.target == 'action':
                    embeddings = get_ss_embeddings(agent, observations, next_observations, mu, std)
                else:
                    embeddings = get_sa_embeddings(agent, observations, actions, mu, std)

                if args.target == 'reward':
                    kf_losses = probe_rewards(embeddings, rewards)
                elif args.target == 'next_obs':
                    kf_losses = probe_next_observations(embeddings, next_observations)
                elif args.target == 'action':
                    kf_losses = probe_actions(embeddings, actions)
 
                res.append((env_name, algo, args.target, seed, step, *kf_losses))
    res_df = pd.DataFrame(res, columns=['env_name', 'algo', 'target', 'seed', 'step',
                                        'cv_loss1', 'cv_loss2', 'cv_loss3', 'cv_loss4',
                                        'cv_loss5'])
    res_df.to_csv(f'probe_exp_res/{args.target}.csv') 


def run_optimal_policy_probing_exp(args):
    res = []
    optimal_td3_agents = get_optimal_td3_agents()
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

        # optimal td3 agent
        optimal_td3_agent = optimal_td3_agents[env_name.split('-')[0]]
        
        # Optimal actions/value functions
        optimal_actions = optimal_td3_agent.select_action(
            optimal_td3_agent.actor_state.params, observations)
        optimal_Q_values = optimal_td3_agent.critic.apply(
            {"params": optimal_td3_agent.critic_state.params},
            observations, actions, method=optimal_td3_agent.critic.Q1)
        optimal_V_values = optimal_td3_agent.critic.apply(
            {"params": optimal_td3_agent.critic_state.params},
            observations, optimal_actions, method=optimal_td3_agent.critic.Q1)

        # probing experiment w.r.t. checkpoints
        for algo in ['td3bc', 'cql', 'combo']:
            agent = AGENTS[algo](obs_dim=obs_dim, act_dim=act_dim)
            for seed in range(10):
                step = baseline_agent_df.loc[(env_name, algo, seed), 'step']

                # Check score before/after training
                agent.load(f"saved_agents/{algo}/{env_name}/s{seed}_{step}")

                if args.target in ['optimal_action', 'optimal_V_value']:
                    embeddings = get_s_embeddings(agent, observations, mu, std)
                else:
                    embeddings = get_sa_embeddings(agent, observations, actions, mu, std)

                if args.target == 'optimal_action':
                    kf_losses = probe_actions(embeddings, optimal_actions)
                elif args.target == 'optimal_Q_value':
                    kf_losses = probe_values(embeddings, optimal_Q_values)
                elif args.target == 'optimal_V_value':
                    kf_losses = probe_values(embeddings, optimal_V_values)
 
                res.append((env_name, algo, args.target, seed, step, *kf_losses))
    res_df = pd.DataFrame(res, columns=['env_name', 'algo', 'target', 'seed', 'step',
                                        'cv_loss1', 'cv_loss2', 'cv_loss3', 'cv_loss4',
                                        'cv_loss5'])
    res_df.to_csv(f'probe_exp_res/{args.target}.csv') 


if __name__ == '__main__':
    args = get_args()
    os.makedirs('probe_exp_res', exist_ok=True)
    # for target in ['action', 'reward', 'next_obs']:
    #     args.target = target
    #     run_dynamics_probing_exp(args)

    for target in ['optimal_action', 'optimal_Q_value', 'optimal_V_value']:
        args.target = target
        run_optimal_policy_probing_exp(args)
