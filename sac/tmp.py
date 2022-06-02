from absl import flags
from gym_utils import make_env
from models import SACAgent
from ml_collections import config_flags
import sys


config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS
FLAGS(sys.argv)
configs = FLAGS.config


env_name = "quadruped-run"
seed = 42
env = make_env(configs.env_name, configs.seed)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]


# SAC agent
agent = SACAgent(obs_dim=obs_dim,
                 act_dim=act_dim,
                 max_action=max_action,
                 seed=configs.seed,
                 tau=configs.tau,
                 gamma=configs.gamma,
                 lr=configs.lr,
                 hidden_dims=configs.hidden_dims,
                 initializer=configs.initializer)


obs, done = env.reset(), False
action = env.action_space.sample()
next_obs, reward, done, info = env.step(action)


agent.rng, action = agent.sample_action(agent.actor_state.params,
                                        agent.rng, obs)
