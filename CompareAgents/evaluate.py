from typing import Tuple
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
import d4rl
import time
import pandas as pd
from tqdm import trange
from models import CDAAgent


def eval_policy(agent: CDAAgent, env: gym.Env, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(agent.agent.actor_state.params, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


# param0 = agent.actor_state.params
expectile = 0.7
temperature = 3.0
for env_name in ["hopper-medium-v2"]:
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    for seed in [0]:
        agent = CDAAgent(env_name=env_name,
                         algo="iql",
                         obs_dim=obs_dim,
                         act_dim=act_dim,
                         expectile=expectile,
                         temperature=temperature)
        exp_name = f"{env_name}/s{seed}"
        res = []
        for step in trange(191, 201):
        # for step in trange(1, 11):
            agent.load(f"saved_models/{exp_name}", step)
            eval_reward, eval_time = eval_policy(agent, env, 10)
            res.append((step, eval_reward, eval_time))
        res_df = pd.DataFrame(res, columns=["step", "reward", "eval_time"])
        print(res_df)
        res_df.to_csv(f"logs/{exp_name}_reward.csv")
