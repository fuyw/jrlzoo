import gym
import torch
from models import DQNAgent, RNDAgent
from utils import ReplayBuffer, register_custom_envs


env_name = "PointmassHard-v2"
register_custom_envs()


env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n


def load_agents():
    rnd_agents, dqn_agents = [], []
    for i in range(1, 21):
        rnd_agent = RNDAgent(lr=1e-3, obs_dim=obs_dim, act_dim=act_dim)
        rnd_agent.load(f"ckpts/rnd{i}.ckpt")
        rnd_agents.append(rnd_agent)
    for i in range(1, 21):
        dqn_agent = DQNAgent(lr=1e-3, obs_dim=obs_dim, act_dim=act_dim)
        dqn_agent.load(f"ckpts/dqn{i}.ckpt")
        dqn_agents.append(dqn_agent)
    return rnd_agents, dqn_agents


def run_trajectory(agent, env):
    obs, done = env.reset(), False
    episode_steps, episode_rewards = 0, 0
    traj_observations = []
    traj_next_observations = []
    traj_actions = []
    traj_rewards = []
    traj_dones = []
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        episode_rewards += reward
        episode_steps += 1
        done_bool = float(done) if episode_steps < env.unwrapped._max_episode_steps else 0

        traj_observations.append(obs)
        traj_next_observations.append(next_obs)
        traj_actions.append(action)
        traj_rewards.append(reward)
        traj_dones.append(done_bool)

        obs = next_obs
    print(f"Run trajectory of len {episode_steps} with reward {episode_rewards}.")
    return traj_observations, traj_actions, traj_rewards, \
        traj_next_observations, traj_dones, episode_rewards


rnd_agents, dqn_agents = load_agents()
rnd_res, dqn_res = [], []
for i in range(20):
    _, _, _, _, _, r = run_trajectory(rnd_agents[i], env)
    rnd_res.append((f"rnd{i+1}", r))

    _, _, _, _, _, r = run_trajectory(dqn_agents[i], env)
    dqn_res.append((f"dqn{i+1}", r))


selected_rnd_agents = [rnd_agents[0], rnd_agents[14], rnd_agents[16]]
selected_dqn_agents = [dqn_agents[1], dqn_agents[2], dqn_agents[3], dqn_agents[5], 
                       dqn_agents[6], dqn_agents[13], dqn_agents[19]]


replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(1e5))


L1, L2 = 47_500, 50_000
while (replay_buffer.size < L1):
    for agent in selected_dqn_agents:
        (traj_observations,
         traj_actions,
         traj_rewards,
         traj_next_observations,
         traj_dones,
         episode_rewards) = run_trajectory(agent, env)
        if episode_rewards < -30:
            for i in range(len(traj_dones)):
                replay_buffer.add(traj_observations[i],
                                  traj_actions[i],
                                  traj_next_observations[i],
                                  traj_rewards[i],
                                  traj_dones[i])

while (replay_buffer.size < L2):
    for agent in selected_rnd_agents:
        (traj_observations,
         traj_actions,
         traj_rewards,
         traj_next_observations,
         traj_dones,
         episode_rewards) = run_trajectory(agent, env)
        for i in range(len(traj_dones)):
            replay_buffer.add(traj_observations[i],
                                traj_actions[i],
                                traj_next_observations[i],
                                traj_rewards[i],
                                traj_dones[i])


replay_buffer.save(f"buffers/pointmass")

# env.plot_trajectory("tmp")