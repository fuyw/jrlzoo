import d4rl
import gym
import time
import pandas as pd
from models import CQLAgent


def eval_policy(agent, eval_env, eval_episodes: int = 10):
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            action = agent.sample_action(obs, True)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


env_names = ["antmaze-umaze-v0", "antmaze-umaze-diverse-v0",
             "antmaze-medium-diverse-v0", "antmaze-medium-play-v0",
             "antmaze-large-diverse-v0", "antmaze-large-play-v0"]
res = []
for env_name in env_names:
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent = CQLAgent(obs_dim=obs_dim,
                     act_dim=act_dim,
                     max_action=max_action)
    agent.load(f"selected_models/{env_name}", 10)
    reward = eval_policy(agent, env, 100)[0]
    res.append((env_name, reward))


#                     env_name  reward
# 0           antmaze-umaze-v0    80.0
# 1   antmaze-umaze-diverse-v0    55.0
# 2  antmaze-medium-diverse-v0    58.0
# 3     antmaze-medium-play-v0    58.0
# 4   antmaze-large-diverse-v0    29.0
# 5      antmaze-large-play-v0    30.0

res_df = pd.DataFrame(res, columns=["env_name", "reward"])
res_df.to_csv("cql_res.csv")
