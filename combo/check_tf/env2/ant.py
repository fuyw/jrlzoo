import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # def __init__(self, goal=15.0/180*np.pi):
    def __init__(self, goal=60.0/180*np.pi):
    # def __init__(self, goal=np.pi):
    # def __init__(self, goal=30.0/180*np.pi):
        self._goal = goal
        self._goals = [45.0/180*np.pi, 60.0/180*np.pi, 90.0/180*np.pi, np.pi]
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        # self.render()
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(a, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        direct = (np.cos(self._goal), np.sin(self._goal))
        directs = [(np.cos(goal), np.sin(goal)) for goal in self._goals]

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # xposbefore = self.get_body_com("torso")[0]
        # self.do_simulation(a, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0]
        
        # forward_reward = (xposafter - xposbefore)/self.dt
        forward_reward = x_velocity
        angle_reward = np.dot(np.array(xy_velocity), direct)
        angle_rewards = [np.dot(np.array(xy_velocity), direct_) for direct_ in directs]
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            reward_forward=forward_reward,
            reward_angle=angle_reward,
            reward_angle_45=angle_rewards[0],
            reward_angle_60=angle_rewards[1],
            reward_angle_90=angle_rewards[2],
            reward_angle_180=angle_rewards[3],
            x_position=xy_position_after[0],
            y_position=xy_position_after[1])

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5