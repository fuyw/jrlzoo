import d4rl
import gym
import jax
import os
import numpy as np
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

from tqdm import trange
from models.combo import COMBOAgent
from models.td3bc import TD3BCAgent
from utils import ReplayBuffer


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


def load_data(args):
    data = np.load(f"saved_buffers/{args.env_name.split('-')[0]}/5agents.npz")
    observations = data["observations"]
    actions = data["actions"]
    next_observations = data["next_observations"]
    rewards = data["rewards"]
    discounts = data["discounts"]
    return observations, actions, rewards, next_observations


def get_embeddings(args, agent, observations, actions):
    if isinstance(agent, TD3BCAgent):
        observations = (observations - mu)/std

    L = len(observations)
    batch_size = 10000
    batch_num = int(np.ceil(L / batch_size))
    encode = jax.jit(agent.encode)
    embeddings = []
    for i in trange(batch_num):
        batch_observations = observations[i*batch_size:(i+1)*batch_size]
        batch_actions = actions[i*batch_size:(i+1)*batch_size]
        batch_embedding = encode(batch_observations, batch_actions)
        embeddings.append(batch_embedding)

    embeddings = np.concatenate(embeddings, axis=0)
    assert len(embeddings) == L
    return embeddings


def probe_rewards(embeddings, rewards):
    from mlp import ProbeTrainer
    trainer = ProbeTrainer(256, 1)
    kf_losses = trainer.train(embeddings, rewards.reshape(-1, 1))
    return kf_losses


for Agent, agent_name, seed in [# (TD3BCAgent, 'td3bc', 0),
                                (COMBOAgent, 'combo', 3)]:
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
