from typing import Tuple

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import d4rl
import numpy as np

import jax
import jax.numpy as jnp
import optax
import time
from flax.training import train_state, checkpoints

from tqdm import trange
from PIL import Image
from models import Actor


def save_video(save_dir, file_name, frames):
    filename = os.path.join(save_dir, file_name)
    if not os.path.exists(filename):
        os.makedirs(filename)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), "RGB")
        img.save(os.path.join(filename, f"frame_{i}.png"))


@jax.jit
def sample_action(actor_state: train_state.TrainState,
                  observation: np.ndarray) -> np.ndarray:
    sampled_action = actor_state.apply_fn({"params": actor_state.params},
                                          observation)
    return sampled_action


def eval_policy(actor_state: train_state.TrainState,
                env: gym.Env,
                eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = sample_action(actor_state, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward, time.time() - t1


for env_name in ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']:
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)

    actor = Actor(act_dim=act_dim)
    actor_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    actor_state = train_state.TrainState.create(apply_fn=actor.apply,
                                                params=actor_params,
                                                tx=optax.adam(0.1))

    eval_reward1, _ = eval_policy(actor_state, env)
    actor_state = checkpoints.restore_checkpoint(
        f'saved_models/{env_name}/s1/actor_100', actor_state)
    eval_reward2, _ = eval_policy(actor_state, env)
    print(f'Reward before training: {eval_reward1:.2f}')
    print(f'Reward after training: {eval_reward2:.2f}')

    # collect one trajectory
    frames = []
    obs, done = env.reset(), False
    curr_frame = env.render(mode="rgb_array")
    curr_frame = curr_frame[::-1]
    frames.append(curr_frame)
    while not done:
        action = sample_action(actor_state, obs)
        frame = env.render('rgb_array')
        obs, reward, done, _ = env.step(action)
        curr_frame = env.render(mode="rgb_array")
        curr_frame = curr_frame[::-1]
        frames.append(curr_frame)
    frames = np.array(frames)

    save_video("saved_video", f"{env_name}", frames)
    fps = 30 if 'HalfCheetah' in env_name else 100
    os.system(
        f"ffmpeg -r {fps} -i saved_video/{env_name}/frame_%01d.png -vcodec mpeg4 -y saved_video/{env_name}.mp4"
    )
