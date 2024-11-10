import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".45"

import imageio

from models import SACAgent
from utils import make_env



def eval_agent(agent,
               eval_env,
               eval_episodes: int = 100,
               save_video: bool = False):
    avg_reward = 0
    avg_step = 0
    avg_success_rate = 0
    frames = []
    for i in range(1, 1+eval_episodes):
        observation, _ = eval_env.reset()
        if save_video and i == eval_episodes:
            frames.append(eval_env.render())
        while True:
            action = agent.sample_action(observation["observation"],
                                         observation["desired_goal"])
            observation, reward, terminal, truncation, info = eval_env.step(action)
            if save_video and i == eval_episodes:
                frames.append(eval_env.render())
            avg_reward += reward
            avg_step += 1
            if terminal or truncation:
                avg_success_rate += info["is_success"]
                break
    avg_reward /= eval_episodes
    avg_step /= eval_episodes
    avg_success_rate /= eval_episodes
    return avg_success_rate, avg_reward, avg_step, frames



env_name = "FetchReach-v3"
max_episode_steps = 100

ckpt_dir = "/home/yuwei/GCRL/fetch_exp/saved_models/sac/FetchReach-v3/N4_L100/s0_20241109_161323"


eval_env = make_env(env_name,
                    num_envs=1,
                    max_episode_steps=max_episode_steps,
                    render_mode="rgb_array")
_ = eval_env.reset()
act_dim = eval_env.action_space.shape[0]
obs_dim = eval_env.observation_space["observation"].shape[0]
goal_dim = eval_env.observation_space["desired_goal"].shape[0]


agent = SACAgent(obs_dim=obs_dim,
                 act_dim=act_dim,
                 goal_dim=goal_dim)

agent.load(ckpt_dir, 5)


avg_success_rate, avg_reward, avg_step, frames = eval_agent(agent, eval_env, 10, save_video=True)
