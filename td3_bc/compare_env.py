import gym
import d4rl
from models import TD3_BC

seed = 0
env_name = 'halfcheetah-medium-v2'

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".1"


env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# random seeds
env.seed(seed)
env.action_space.seed(seed)
np.random.seed(seed)


agent = TD3_BC(obs_dim=obs_dim,
               act_dim=act_dim,
               max_action=max_action)

def eval_policy(agent: TD3_BC,
                env_name: str,
                seed: int,
                mean: np.ndarray,
                std: np.ndarray,
                eval_episodes: int = 10) -> float:
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        t = 0
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            obs = (np.array(obs).reshape(1, -1) - mean) / std
            action = agent.select_action(agent.actor_state.params, obs)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score
