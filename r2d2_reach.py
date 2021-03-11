import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from modify_xml import modify_r2d2
import random as rand

#policy -> 0.8 0.5
#policy2 -> 0.8 0.5 lr 1e-4 tau 0.005 memory 1e6
#policy3 -> -0.8 -0.5 lr 1e-3 tau 0.005 memory 1e5
#policy4 -> new policy random points (0.6 1) lr 1e-4 tau 0.01 memory 1e6
#policy5 -> new policy 0.8 0.5, lr 1e-4 memory 1e5 tau 0.01 ok
#policy6 -> new policy random (0 1) points lr 1e-4 tau 0.01 memory 1e6

radius = 1
x_range = np.arange(0.6, radius, 0.2)

class r2d2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.points = []
        for x in range(x_range.__len__()):
            y = np.sqrt(radius ** 2 - x ** 2)
            self.points.append([x, y])
        self.y_target = np.random.choice(x_range, 1).item()
        self.x_target = np.sqrt(radius**2 - self.y_target**2)
        # self.x_target = 0.8
        # self.y_target = 0.5
        modify_r2d2(self.x_target, self.y_target)
        self.xposbefore = 0
        self.yposbefore = 0
        self.xposafter = 0
        self.yposafter = 0
        self.d0 = np.sqrt(self.x_target**2 + self.y_target**2)
        mujoco_env.MujocoEnv.__init__(self, '/home/niroantonio/PycharmProjects/pythonProject5/r2d22.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.xposbefore = self.get_body_com("torso")[0]
        self.yposbefore = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        self.xposafter = self.get_body_com("torso")[0]
        self.yposafter = self.get_body_com("torso")[1]
        dist = np.sqrt((self.x_target - self.xposafter)**2 + (self.y_target - self.yposafter)**2)
        #dist = np.abs(self.x_target - self.xposafter)
        ob = self._get_obs()
        vel = self.get_vel()
        done = False
        reward = -dist
        if dist <= 0.1:
            reward = 1 - sum(np.abs(self.sim.data.qvel[-4:]))
        if dist <= 0.05:
            reward = 2 - sum(np.abs(self.sim.data.qvel[-4:]))
            # if vel <= 0.005:
            #     done = True
            #     reward += 100
        if self.get_body_com("front_left_wheel")[2] > 0.1 or self.get_body_com("front_right_wheel")[2] > 0.1 or self.get_body_com("back_left_wheel")[2] > 0.1 or self.get_body_com("back_right_wheel")[2] > 0.1:
            done = True
            reward -= 100
        return ob, reward, done, {}

    def get_vel(self):
        xvel = (self.xposafter - self.xposbefore)/self.dt
        yvel = (self.yposafter - self.yposbefore)/self.dt
        vel = np.sqrt(xvel**2 + yvel**2)
        return vel

    def _get_obs(self):
        vel = self.get_vel()
        x = self.get_body_com("torso")[0]
        y = self.get_body_com("torso")[1]
        dist = np.sqrt((self.x_target - x)**2 + (self.y_target - y)**2)
        wheel_vel = self.sim.data.qvel[-4:]
        ob = [vel, dist, self.get_body_com("torso")[0], self.get_body_com("torso")[1], self.x_target, self.y_target]
        ob = np.array(ob)
        ob = np.concatenate([wheel_vel, ob])
        return ob

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        self.xposbefore = 0
        self.yposbefore = 0
        self.xposafter = 0
        self.yposafter = 0
        # self.x_target = 0.8
        # self.y_target = 0.5
        # self.y_target = np.random.choice(x_range,1).item()
        # self.x_target = np.sqrt(radius**2 - self.y_target**2)
        self.d0 = np.sqrt(self.x_target ** 2 + self.y_target ** 2)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

