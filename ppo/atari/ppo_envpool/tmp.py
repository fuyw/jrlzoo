import env_utils

env = env_utils.create_env("PongNoFrameskip-v4", clip_rewards=True)
obs1 = env.reset()  # (84, 84, 4)
obs2, _, _, _ = env.step(0)
