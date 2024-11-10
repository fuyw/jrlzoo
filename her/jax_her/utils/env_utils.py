import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


def make_env(env_name: str = "FetchReach-v3",
             num_envs: int = 1,
             max_episode_steps: int = 100,
             render_mode: str = None):
    if num_envs > 1:
        env = gym.make_vec(env_name,
                           num_envs=num_envs,
                           max_episode_steps=max_episode_steps)
    else:
        env = gym.make(env_name,
                       render_mode=render_mode,
                       max_episode_steps=max_episode_steps)
    return env
