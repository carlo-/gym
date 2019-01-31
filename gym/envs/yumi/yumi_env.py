import os
from pathlib import Path

import mujoco_py
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from gym.envs.robotics.utils import ctrl_set_action


def body_index(model, body_name):
    return model.body_names.index(body_name)


def body_pos(model, body_name):
    ind = body_index(model, body_name)
    return model.body_pos[ind]


def body_quat(model, body_name):
    ind = body_index(model, body_name)
    return model.body_quat[ind]


def body_frame(env, body_name):
    """
    Returns the rotation matrix to convert to the frame of the named body
    """
    ind = body_index(env.model, body_name)
    b = env.data.body_xpos[ind]
    q = env.data.body_xquat[ind]
    qr, qi, qj, qk = q
    s = np.square(q).sum()
    R = np.array([
        [1 - 2 * s * (qj ** 2 + qk ** 2), 2 * s * (qi * qj - qk * qr), 2 * s * (qi * qk + qj * qr)],
        [2 * s * (qi * qj + qk * qr), 1 - 2 * s * (qi ** 2 + qk ** 2), 2 * s * (qj * qk - qi * qr)],
        [2 * s * (qi * qk - qj * qr), 2 * s * (qj * qk + qi * qr), 1 - 2 * s * (qi ** 2 + qj ** 2)]
    ])
    return R


class YumiEnv(mujoco_env.MujocoEnv):

    # noinspection PyMissingConstructor
    def __init__(self, arm='right', block_gripper=False):

        if arm not in ['right', 'left', 'both']:
            raise ValueError
        self.arm = arm

        self.block_gripper = block_gripper
        self._gripper_r_joint_idx = None
        self._gripper_l_joint_idx = None
        self._arm_r_joint_idx = None
        self._arm_l_joint_idx = None

        self._gripper_joint_max = 0.02
        high = np.array([40, 35, 30, 20, 15, 10, 10]) * 10
        if not block_gripper:
            high = np.r_[high, self._gripper_joint_max]
        if arm == 'both':
            high = np.r_[high, high]

        self.high = high
        self.low = -self.high
        self.wt = 0.9
        self.we = 1 - self.wt

        ################################### Mujoco env init ###################################
        root_dir = os.path.dirname(__file__)
        xml_path = os.path.join(root_dir, 'assets', f'yumi_{arm}.xml')
        if not Path(xml_path).exists():
            raise IOError(f'File {xml_path} does not exist')
        self.frame_skip = 1
        self.model = mujoco_py.load_model_from_path(xml_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        #######################################################################################

        yumi_arm_joints = [1, 2, 7, 3, 4, 5, 6]
        if self.has_right_arm:
            self._arm_r_joint_idx = [self.sim.model.joint_name2id(f'yumi_joint_{i}_r') for i in yumi_arm_joints]
        if self.has_left_arm:
            self._arm_l_joint_idx = [self.sim.model.joint_name2id(f'yumi_joint_{i}_l') for i in yumi_arm_joints]

        if not block_gripper:
            if self.has_right_arm:
                self._gripper_r_joint_idx = [self.sim.model.joint_name2id('gripper_r_joint'),
                                             self.sim.model.joint_name2id('gripper_r_joint_m')]
            if self.has_left_arm:
                self._gripper_l_joint_idx = [self.sim.model.joint_name2id('gripper_l_joint'),
                                             self.sim.model.joint_name2id('gripper_l_joint_m')]

        self.obs_dim = self._get_obs().size
        self.action_space = spaces.Box(low=-np.ones(high.size), high=np.ones(high.size), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.seed()

    @property
    def has_right_arm(self):
        return self.arm == 'right' or self.arm == 'both'

    @property
    def has_left_arm(self):
        return self.arm == 'left' or self.arm == 'both'

    @property
    def has_two_arms(self):
        return self.arm == 'both'

    @property
    def _gripper_base(self):
        r_base = 'gripper_r_base'
        l_base = 'gripper_l_base'
        if self.arm == 'both':
            return l_base, r_base
        elif self.arm == 'right':
            return r_base
        else:
            return l_base

    def _set_action(self, a):
        a = np.clip(a, self.action_space.low, self.action_space.high)
        a *= self.high

        if not self.block_gripper:
            arm1_a = a[:8]
            arm2_a = a[8:]
            gripper1_a = arm1_a[7:]
            gripper2_a = arm2_a[7:]
            a = np.r_[arm1_a, gripper1_a, arm2_a, gripper2_a]
        else:
            arm1_a = a[:7]
            arm2_a = a[7:]
            g = self._gripper_joint_max
            a = np.r_[arm1_a, g, g]
            if self.has_two_arms:
                a = np.r_[a, arm2_a, g, g]

        ctrl_set_action(self.sim, a)
        self.sim.step()
        return a

    def step(self, a):
        a_real = self._set_action(a)
        if self.arm == 'both':
            reward = self._reward(a_real, self._gripper_base[0]) + self._reward(a_real, self._gripper_base[1])
        else:
            reward = self._reward(a_real)
        return self._get_obs(), reward, False, {}

    def _reward(self, a, gripper_base=None):

        gripper_base = gripper_base or self._gripper_base
        assert isinstance(gripper_base, str)
        eef = self.get_body_com(gripper_base)

        goal = self.get_body_com('goal')
        goal_distance = np.linalg.norm(eef - goal)
        # This is the norm of the joint angles
        # The ** 4 is to create a "flat" region around [0, 0, 0, ...]
        q_norm = np.linalg.norm(self.sim.data.qpos.flat[:7]) ** 4 / 100.0

        # TODO in the future
        # f_desired = np.eye(3)
        # f_current = body_frame(self, 'gripper_r_base')

        reward = -(
            self.wt * goal_distance * 2.0 +  # Scalars here is to make this part of the reward approx. [0, 1]
            self.we * np.linalg.norm(a) / 40 +
            q_norm
        )
        return reward

    def _get_obs(self):

        arm_l_qpos = np.zeros(0)
        arm_l_qvel = np.zeros(0)
        gripper_l_qpos = np.zeros(0)

        arm_r_qpos = np.zeros(0)
        arm_r_qvel = np.zeros(0)
        gripper_r_qpos = np.zeros(0)

        if self._arm_l_joint_idx is not None:
            arm_l_qpos = self.sim.data.qpos[self._arm_l_joint_idx]
            arm_l_qvel = self.sim.data.qvel[self._arm_l_joint_idx]
            arm_l_qvel = np.clip(arm_l_qvel, -10, 10)

        if self._gripper_l_joint_idx is not None:
            gripper_l_qpos = self.sim.data.qpos[self._gripper_l_joint_idx]

        if self._arm_r_joint_idx is not None:
            arm_r_qpos = self.sim.data.qpos[self._arm_r_joint_idx]
            arm_r_qvel = self.sim.data.qvel[self._arm_r_joint_idx]
            arm_r_qvel = np.clip(arm_r_qvel, -10, 10)

        if self._gripper_r_joint_idx is not None:
            gripper_r_qpos = self.sim.data.qpos[self._gripper_r_joint_idx]

        obs = np.concatenate([
            arm_l_qpos, arm_l_qvel, gripper_l_qpos,
            arm_r_qpos, arm_r_qvel, gripper_r_qpos
        ])

        return obs

    def reset_model(self):
        pos_low = np.r_[-1.0, -0.3, -0.4, -0.4, -0.3, -0.3, -0.3]
        pos_high = np.r_[0.4,  0.6,  0.4,  0.4,  0.3,  0.3,  0.3]
        vel_high = np.ones(7) * 0.5
        vel_low = -vel_high

        if self._arm_l_joint_idx is not None:
            self.init_qpos[self._arm_l_joint_idx] = np.random.uniform(pos_low, pos_high)
            self.init_qvel[self._arm_l_joint_idx] = np.random.uniform(vel_low, vel_high)

        if self._gripper_l_joint_idx is not None:
            self.init_qpos[self._gripper_l_joint_idx] = 0.0
            self.init_qvel[self._gripper_l_joint_idx] = 0.0

        if self._arm_r_joint_idx is not None:
            self.init_qpos[self._arm_r_joint_idx] = np.random.uniform(pos_low, pos_high)
            self.init_qvel[self._arm_r_joint_idx] = np.random.uniform(vel_low, vel_high)

        if self._gripper_r_joint_idx is not None:
            self.init_qpos[self._gripper_r_joint_idx] = 0.0
            self.init_qvel[self._gripper_r_joint_idx] = 0.0

        self.init_qpos *= 0.0
        self.init_qvel *= 0.0

        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 2.0
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 180


class YumiRightArmEnv(YumiEnv, utils.EzPickle):
    def __init__(self):
        super().__init__(arm='right')
        utils.EzPickle.__init__(self)


class YumiLeftArmEnv(YumiEnv, utils.EzPickle):
    def __init__(self):
        super().__init__(arm='left')
        utils.EzPickle.__init__(self)


class YumiTwoArmsEnv(YumiEnv, utils.EzPickle):
    def __init__(self):
        super().__init__(arm='both')
        utils.EzPickle.__init__(self)
