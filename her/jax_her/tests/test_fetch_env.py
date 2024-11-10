import gymnasium as gym
import gymnasium_robotics
import numpy as np
gym.register_envs(gymnasium_robotics)


env_name = "FetchReach-v3"
env = gym.make(env_name)

obs, _ = env.reset()
observations = [obs["observation"]]
achieved_goals = [obs["achieved_goal"]]
desired_goals = [obs["desired_goal"]]
rewards = []
terminals = []
truncations = []

while True:
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    observations.append(obs["observation"])
    achieved_goals.append(obs["achieved_goal"])
    desired_goals.append(obs["desired_goal"])
    rewards.append(reward)
    terminals.append(terminated)
    truncations.append(truncated)
    if terminated or truncated:
        break

observations = np.array(observations)      # 115.03999409314719
achieved_goals = np.array(achieved_goals)  # 115.12862632403018
desired_goals = np.array(desired_goals)    # 131.82771454124287
rewards = np.array(rewards)
