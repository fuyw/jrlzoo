import gym
from gym.wrappers import RescaleAction
from gym.wrappers.pixel_observation import PixelObservationWrapper

import wrappers


def make_env(env_name: str,
             seed: int,
             action_repeat: int = 1,
             num_stack: int = 1,
             image_size: int = 84,
             from_pixels: bool = False):
    domain_name, task_name = env_name.split("-")
    env = wrappers.DMCEnv(domain_name=domain_name,
                          task_name=task_name,
                          task_kwargs={"random": seed})

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if from_pixels:
        if "quadruped" in env_name:
            camera_id = 2
        else:
            camera_id = 0

        env = PixelObservationWrapper(env,
                                      pixels_only=True,
                                      render_kwargs={
                                        "pixels": {
                                            "height": image_size,
                                            "width": image_size,
                                            "camera_id": camera_id,
                                        }
                                      })

    if num_stack > 1:
        env = wrappers.FrameStack(env, num_stack=num_stack)

    env = gym.wrappers.ClipAction(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
