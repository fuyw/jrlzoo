from gym.envs.registration import register

register(
    id='obstacles-cs285-v0',
    entry_point='envs.obstacles:Obstacles',
    max_episode_steps=500,
)
from envs.obstacles.obstacles_env import Obstacles
