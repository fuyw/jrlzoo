from atari_utils import create_env


env_name = "BreakoutNoFrameskip-v4"
env = create_env(env_name)
print(f"obs_shape = {env.reset().shape}")  # (84, 84)



