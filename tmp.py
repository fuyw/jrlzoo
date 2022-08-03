import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
import numpy as np
import gym
import d4rl
import jax.numpy as jnp

from td3.models import TD3Agent
from sac.models import SACAgent
from td3.utils import ReplayBuffer

task = "walker2d"
td3_ckpt_dirs = [i for i in os.listdir(f"td3/saved_models/{task}-v2") if "td3_s" in i]
sac_ckpt_dirs = [i for i in os.listdir(f"sac/saved_models/{task}-v2") if "sac_s" in i]
env_name = f"{task}-medium-v2"
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
replay_buffer = ReplayBuffer(obs_dim, act_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

check_observations = replay_buffer.observations[:50_000]
check_actions = replay_buffer.actions[:50_000]
td3_agent = TD3Agent(obs_dim, act_dim, 1.0)
td3_agent.load(f"td3/saved_models/{task}-v2/{td3_ckpt_dirs[0]}", 10)
td3_Q1, td3_Q2 = td3_agent.critic.apply({"params": td3_agent.critic_state.params}, check_observations, check_actions)
td3_Q = jnp.minimum(td3_Q1, td3_Q2)

sac_agent = SACAgent(obs_dim, act_dim, 1.0)
sac_agent.load(f"sac/saved_models/{task}-v2/{sac_ckpt_dirs[0]}", 10)
sac_Q1, sac_Q2 = sac_agent.critic.apply({"params": sac_agent.critic_state.params}, check_observations, check_actions)
sac_Q = jnp.minimum(sac_Q1, sac_Q2)

tmp = 2 * np.abs(td3_Q - sac_Q) / (np.abs(td3_Q) + np.abs(sac_Q))
print(tmp.mean())
