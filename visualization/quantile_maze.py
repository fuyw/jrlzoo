# -*- coding: utf-8 -*-
"""quantile_maze.ipynb
Original file is located at
    https://colab.research.google.com/drive/1s2kS-pDoo2EIdL1YmxGzV8ai_7dunGWu

pip install git+https://github.com/zuoxingdong/mazelab.git
pip install -q mediapy


~/miniforge3/lib/python3.9/site-packages/mazelab/generators/morris_water_maze.py in <module>
      1 import numpy as np
      2 
----> 3 from skimage.draw import circle
"""

from mazelab.generators import u_maze
from mazelab.solvers import dijkstra_solver
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color
from mazelab import BaseEnv
from mazelab import VonNeumannMotion

import flax
from flax.training.train_state import TrainState

import numpy as np
import jax.numpy as jnp
import optax
import jax

import gym
from gym.spaces import Box
from gym.spaces import Discrete
import mediapy as media
import copy

x = u_maze(width=10, height=9, obstacle_width=6, obstacle_height=3)

start_idx = [[8, 1]]
goal_idx = [[1, 1]]
env_id = "UMaze-v0"


class Maze(BaseMaze):
    @property
    def size(self):
        return x.shape

    def make_objects(self):
        free = Object(name="free",
                      value=0,
                      rgb=color.free,
                      impassable=False,
                      positions=np.stack(np.where(x == 0), axis=1))
        obstacle = Object(name="obstacle",
                          value=1,
                          rgb=color.obstacle,
                          impassable=True,
                          positions=np.stack(np.where(x == 1), axis=1))
        agent = Object(name="agent",
                       value=2,
                       rgb=color.agent,
                       impassable=False,
                       positions=[])
        goal = Object(name="goal",
                      value=3,
                      rgb=color.goal,
                      impassable=False,
                      positions=[])
        return free, obstacle, agent, goal


class Env(BaseEnv):
    def __init__(self):
        super().__init__()
        self.maze = Maze()
        self.motions = VonNeumannMotion()
        self.observation_space = Box(low=0,
                                     high=len(self.maze.objects),
                                     shape=self.maze.size,
                                     dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

    def step(self, action):
        if np.random.uniform() < 0.25:
            action = np.random.randint(0, 4)
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [
            current_position[0] + motion[0], current_position[1] + motion[1]
        ]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +10
            done = True
        else:
            # reward = np.random.uniform(0.0, 0.2)
            reward = 0.0
            done = False
        return self.maze.to_value(), reward, done, {}

    def reset(self):
        self.maze.objects.agent.positions = start_idx
        self.maze.objects.goal.positions = goal_idx
        return self.maze.to_value()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[
            1] < self.maze.size[1]
        if (nonnegative and within_edge):
            passable = not self.maze.to_impassable()[position[0]][position[1]]
        else:
            passable = False
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


images = []

states = []
actions = []
rewards = []
next_states = []
masks = []

TIMELIMIT = 100

mix = 0.1
for i in range(100):
    j = 0
    done = False
    env = Env()
    env.reset()
    impassable = env.unwrapped.maze.to_impassable()
    while not done and j < TIMELIMIT:
        obs = np.copy(env.maze.objects.agent.positions[0])
        action = dijkstra_solver(impassable, env.motions,
                                 env.maze.objects.agent.positions[0],
                                 env.maze.objects.goal.positions[0])[0]
        if i > 0:
            while True:
                tmp_action = env.action_space.sample()
                if True or np.random.uniform(
                        0.0, 1.0) < 0.5 or tmp_action != action:
                    action = tmp_action
                    break
        _, reward, done, info = env.step(action)
        next_obs = np.copy(env.maze.objects.agent.positions[0])
        j += 1
        array = env.render(mode="rgb_array")
        images.append(array)

        states.append(obs)
        actions.append(np.copy(action))
        next_states.append(next_obs)
        rewards.append(np.copy(reward))
        masks.append(1.0 - float(done))

states = np.stack(states)
actions = np.stack(actions)
next_states = np.stack(next_states)
rewards = np.stack(rewards)
masks = np.stack(masks)

media.show_video(images,
                 width=images[0].shape[1],
                 height=images[0].shape[0],
                 fps=10)

q = {"w": jnp.zeros((*impassable.shape, 4), dtype=jnp.float32)}
q_target = {"w": jnp.zeros((*impassable.shape, 4), dtype=jnp.float32)}
v = {"w": jnp.zeros(impassable.shape, dtype=jnp.float32)}

optimizer = optax.adam(1e-3)
q_opt_state = optimizer.init(q)
v_opt_state = optimizer.init(v)


@jax.jit
def fit_q(batch, q, v, q_opt_state):
    vs = jax.vmap(lambda y: v["w"][y[0], y[1]])(batch["next_states"])
    targets = batch["rewards"] + 0.9 * vs * batch["masks"]

    def q_loss(q_params):
        qs = jax.vmap(lambda s, a: q_params["w"][s[0], s[1], a])(
            batch["states"], batch["actions"])
        return ((targets - qs)**2).mean()

    loss, grads = jax.value_and_grad(q_loss)(q)

    updates, opt_state = optimizer.update(grads, q_opt_state)
    new_q = optax.apply_updates(q, updates)
    return loss, new_q, opt_state


def quantile_loss(diff, quantile=0.5):
    # return diff ** 2
    weight = jnp.where(diff > 0, quantile, (1 - quantile))
    soft_l1_loss = jnp.where(
        jnp.abs(diff) > 1.0,
        jnp.abs(diff) - 0.5, 0.5 * diff**2)
    return weight * (diff)**2


@jax.jit
def fit_v(batch, q, v, v_opt_state):
    qs = jax.vmap(lambda y, z: q["w"][y[0], y[1], z])(batch["states"],
                                                      batch["actions"])

    def v_loss(v_params):
        vs = jax.vmap(lambda s: v_params["w"][s[0], s[1]])(batch["states"])
        return (quantile_loss(qs - vs)).mean()

    loss, grads = jax.value_and_grad(v_loss)(v)

    updates, opt_state = optimizer.update(grads, v_opt_state)
    new_v = optax.apply_updates(v, updates)
    return loss, new_v, opt_state


for i in range(100000):
    indx = np.random.randint(0, len(states), size=1024)
    batch = {
        "states": states[indx],
        "actions": actions[indx],
        "next_states": next_states[indx],
        "rewards": rewards[indx],
        "masks": masks[indx]
    }
    q_loss, q, q_opt_state = fit_q(batch, q, v, q_opt_state)
    v_loss, v, v_opt_state = fit_v(batch, q_target, v, v_opt_state)
    if i % 100 == 0:
        q_target = copy.deepcopy(q)
    if i % 10000 == 0:
        print(q_loss, v_loss)

np.set_printoptions(precision=2,
                    floatmode="fixed",
                    suppress=True,
                    linewidth=10000)

tmp_v = np.array(v["w"]).reshape(impassable.shape)
print(np.array(q["w"].argmax(-1)).reshape(impassable.shape))

# !pip install matplotlib==3.1.3

# Commented out IPython magic to ensure Python compatibility.
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
# plt.style.use("bmh")
plt.style.use("tableau-colorblind10")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["legend.fontsize"] = 10
# plt.rcParams["legend.loc"] = "lower right"
plt.rcParams["figure.facecolor"] = "#FFFFFF"

q_actions = np.array(q["w"].argmax(-1)).reshape(impassable.shape)
plt.imshow(tmp_v, cmap="cividis")
for i in range(q_actions.shape[0]):
    for j in range(q_actions.shape[1]):
        if q_actions[i, j] == 0:
            di = -1.0
            dj = 0.0
        elif q_actions[i, j] == 1:
            di = 1.0
            dj = 0.0
        elif q_actions[i, j] == 2:
            di = 0.0
            dj = -1.0
        else:
            di = 0.0
            dj = 1.0
        scaling = 0.1
        if not impassable[i, j] and tmp_v[i, j] > 0:
            plt.arrow(j - dj * scaling * 3,
                      i - di * scaling * 3,
                      dj * scaling,
                      di * scaling,
                      fc="k",
                      ec="k",
                      head_width=0.5,
                      head_length=0.5)

plt.xticks([])
plt.yticks([])
plt.clim(0, 10)
plt.tight_layout()
# plt.colorbar()
plt.savefig("maze_2.pdf")

import jax.numpy as jnp


def quantile_loss(diff, quantile=0.95):
    weight = jnp.where(diff > 0, quantile, (1 - quantile))
    return weight * (diff)**2
