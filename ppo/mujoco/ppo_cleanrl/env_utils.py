import gym


def create_env(env_name: str, seed: int):
    def thunk():
        env = gym.make(env_name)
        env.seed(seed)
        env.action_space.seed(seed)
        return env
    return thunk
