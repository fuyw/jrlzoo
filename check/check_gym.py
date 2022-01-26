import gym
import d4rl
from models import TD3
from utils import ReplayBuffer


TASKS = ['HalfCheetah', 'Hopper', 'Walker2d']


for task in TASKS:
    env_name = f'{task.lower()}-medium-v2'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent = TD3(obs_dim=obs_dim, act_dim=act_dim, max_action=max_action)
    agent.load(f"saved_models/{task}-v2/step100_seed1")
    replay_buffer = ReplayBuffer(obs_dim, act_dim)

    state_lst = []
    for _ in range(5):
        obs, done = env.reset(), False
        while not done:
            state_lst.append(env.sim.get_state())
            action = agent.select_action(agent.actor_state.params, np.array(obs))
            next_obs, reward, done, _ = env.step(action)
            transition = (obs, action, reward, next_obs, done)
            replay_buffer.add(obs, action, next_obs, reward, done)
            obs = next_obs

    print(len(state_lst))
    print(replay_buffer.size)

    for idx in range(5000):
        obs = env.reset()
        state_obs = env.sim.get_state()
        state_obs.qpos[:] = state_lst[idx].qpos
        state_obs.qvel[:] = state_lst[idx].qvel
        env.sim.set_state(state_obs)
        next_obs, reward, done, _ = env.step(replay_buffer.actions[idx])

        delta_next_obs = abs(replay_buffer.next_observations[idx] - next_obs).sum()
        delta_reward = abs(replay_buffer.rewards[idx] - reward).sum()

        assert delta_next_obs < 1e-6
        assert delta_reward < 1e-6
