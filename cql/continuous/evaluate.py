from typing import Tuple
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
import d4rl
import time
import pandas as pd
from tqdm import trange
from models import CQLAgent


def eval_policy(agent: CQLAgent, env: gym.Env, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(agent.actor_state.params, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


# param0 = agent.actor_state.params
for env_name in ["antmaze-medium-diverse-v0",
                 "antmaze-medium-play-v0",
                 "antmaze-large-play-v0",
                 "antmaze-large-diverse-v0"]:
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    for num_layers in [2, 3]:
        seed = 0
        agent = CQLAgent(obs_dim=obs_dim, act_dim=act_dim,
                        actor_hidden_dims=tuple([256]*num_layers),
                        critic_hidden_dims=tuple([256]*num_layers))
        exp_name = f"{env_name}/cql_s{seed}_L{num_layers}{num_layers}_Lag1_MTB1_offset"
        res = []
        for step in trange(1, 11):
            agent.load(f"saved_models/{exp_name}", step*10)
            eval_reward, eval_time = eval_policy(agent, env, 100)
            res.append((eval_reward, eval_time))
        res_df = pd.DataFrame(res, columns=["reward", "eval_time"])
        print(res_df)
        res_df.to_csv(f"logs/{exp_name}_reward.csv")
