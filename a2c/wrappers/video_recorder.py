import os

import gym
import imageio
import numpy as np

from wrappers.common import TimeStep


# Take from
# https://github.com/denisyarats/pytorch_sac/
class VideoRecorder(gym.Wrapper):

    def __init__(self,
                 env: gym.Env,
                 save_folder: str = '',
                 height: int = 480,
                 width: int = 480,
                 fps: int = 60):
        super().__init__(env)

        self.current_episode = 0
        self.save_folder = save_folder
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

        try:
            os.makedirs(save_folder, exist_ok=True)
        except:
            pass

    def step(self, action: np.ndarray) -> TimeStep:

        frame = self.env.render(mode='rgb_array',
                                height=self.height,
                                width=self.width)

        if frame is None:
            try:
                frame = self.sim.render(width=self.width,
                                        height=self.height,
                                        mode='offscreen')
                frame = np.flipud(frame)
            except:
                raise NotImplementedError('Rendering is not implemented.')

        self.frames.append(frame)

        observation, reward, done, info = self.env.step(action)

        if done:
            self.save_video()

        return observation, reward, done, info

    def save_video(self):
        save_file = os.path.join(self.save_folder,
                                    f'{self.current_episode}.mp4')
        imageio.mimsave(save_file, self.frames, fps=self.fps)
        self.frames = []
        self.current_episode += 1