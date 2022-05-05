import os
os.environ['VISIBLE_CUDA_DEVICES']='-1'
import gym
import d4rl
from models import TD3, TD3_BC
AGENT_DICT = {'offline_td3': TD3, 'td3bc': TD3_BC}

for algo in ['offline_td3']:
    for env_name in ['hopper-medium-v2']:
        env = gym.make(env_name)
        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        agent = AGENT_DICT[algo](obs_dim, act_dim)
        agent.load(f'saved_models/{algo}/{env_name}/s0_200')

