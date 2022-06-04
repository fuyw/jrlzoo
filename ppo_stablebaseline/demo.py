import gym

from stable_baselines3.common.policies improt MlpPolicy
from stable_baselines3.common import make_vec_env
from stable_baselines3 import PPO, PPO2


def run():
    env = gym.make('CartPole-v1')

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10_000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()


def run_vec_env():
    # multiprocess environment
    env = make_vec_env('CartPole-v1', n_envs=4)

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save('ppo2_cartpole')

    del model  # remove to demonstrate saving and loading

    model = PPO2.load('ppo2_cartpole')

    # use the trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, _ = env.step(action)
        env.render()

