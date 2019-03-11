import os

import mujoco_py
import numpy as np
from gym.utils import EzPickle
from gym.envs.robotics.robot_env import RobotEnv
from gym.envs.robotics.utils import ctrl_set_action


class YumiEnv(RobotEnv):

    def __init__(self, *, arm, block_gripper, reward_type, distance_threshold, has_object):

        if arm not in ['right', 'left', 'both']:
            raise ValueError
        self.arm = arm

        if reward_type not in ['sparse', 'dense']:
            raise ValueError
        self.reward_type = reward_type

        self.block_gripper = block_gripper
        self.has_object = has_object
        self.distance_threshold = distance_threshold

        self._table_safe_bounds = (np.r_[-0.27, -0.33], np.r_[0.17, 0.33])
        self._reach_bounds_l = (np.r_[-0.27, -0.05, 0.], np.r_[0.17, 0.33, 0.6])
        self._reach_bounds_r = (np.r_[-0.27, -0.33, 0.], np.r_[0.17, 0.05, 0.6])
        self._target_bounds_l = (np.r_[-0.27, 0.05, 0.05], np.r_[0.17, 0.33, 0.4])
        self._target_bounds_r = (np.r_[-0.27, -0.28, 0.05], np.r_[0.17, 0.05, 0.4])

        self._gripper_r_joint_idx = None
        self._gripper_l_joint_idx = None
        self._arm_r_joint_idx = None
        self._arm_l_joint_idx = None

        self._gripper_joint_max = 0.02
        ctrl_high = np.array([40, 35, 30, 20, 15, 10, 10]) * 10
        if not block_gripper:
            ctrl_high = np.r_[ctrl_high, self._gripper_joint_max]
        if arm == 'both':
            ctrl_high = np.r_[ctrl_high, ctrl_high]

        self._ctrl_high = ctrl_high
        self._ctrl_low = -ctrl_high

        model_path = os.path.join(os.path.dirname(__file__), 'assets', f'yumi_{arm}.xml')
        super(YumiEnv, self).__init__(model_path=model_path, n_substeps=1,
                                      n_actions=ctrl_high.size, initial_qpos=None)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict):
        assert achieved_goal.shape == desired_goal.shape
        if self.reward_type == 'sparse':
            if self.has_object:
                raise NotImplementedError
            else:
                rew = float(self.has_left_arm) * self._is_success(achieved_goal[..., :3], desired_goal[..., :3])
                rew += float(self.has_right_arm) * self._is_success(achieved_goal[..., 3:], desired_goal[..., 3:])
                return rew
        else:
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return -d

    # RobotEnv methods
    # ----------------------------

    def _reset_sim(self):
        pos_low = np.r_[-1.0, -0.3, -0.4, -0.4, -0.3, -0.3, -0.3]
        pos_high = np.r_[0.4,  0.6,  0.4,  0.4,  0.3,  0.3,  0.3]
        vel_high = np.zeros(7)
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

        # self.init_qpos *= 0.0
        # self.init_qvel *= 0.0

        self._set_sim_state(self.init_qpos, self.init_qvel)
        return True

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

        return {
            'observation': obs,
            'achieved_goal': self._get_achieved_goal(),
            'desired_goal': self.goal.copy(),
        }

    def _set_action(self, a):
        a = np.clip(a, self.action_space.low, self.action_space.high)
        a *= self._ctrl_high

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
        return a

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        if self.has_object:
            # Goal is object target position
            raise NotImplementedError
        else:
            # Goal is gripper(s) target position(s)
            new_goal = np.zeros(6)
            if self.has_left_arm:
                new_goal[:3] = self.np_random.uniform(*self._target_bounds_l)
            if self.has_right_arm:
                new_goal[3:] = self.np_random.uniform(*self._target_bounds_r)
        return new_goal

    def _env_setup(self, initial_qpos):
        if initial_qpos is not None:
            raise NotImplementedError

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        yumi_arm_joints = [1, 2, 7, 3, 4, 5, 6]
        if self.has_right_arm:
            self._arm_r_joint_idx = [self.sim.model.joint_name2id(f'yumi_joint_{i}_r') for i in yumi_arm_joints]
        if self.has_left_arm:
            self._arm_l_joint_idx = [self.sim.model.joint_name2id(f'yumi_joint_{i}_l') for i in yumi_arm_joints]

        if not self.block_gripper:
            if self.has_right_arm:
                self._gripper_r_joint_idx = [self.sim.model.joint_name2id('gripper_r_joint'),
                                             self.sim.model.joint_name2id('gripper_r_joint_m')]
            if self.has_left_arm:
                self._gripper_l_joint_idx = [self.sim.model.joint_name2id('gripper_l_joint'),
                                             self.sim.model.joint_name2id('gripper_l_joint_m')]

        # Extract information for sampling goals.
        if self.has_left_arm:
            self._initial_l_gripper_pos = self.sim.data.get_site_xpos('gripper_l_center').copy()
        if self.has_right_arm:
            self._initial_r_gripper_pos = self.sim.data.get_site_xpos('gripper_r_center').copy()

    def _viewer_setup(self):
        self.viewer.cam.distance = 2.0
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 180

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        if self.has_left_arm:
            site_id = self.sim.model.site_name2id('target_l')
            self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[site_id]
        if self.has_right_arm:
            site_id = self.sim.model.site_name2id('target_r')
            self.sim.model.site_pos[site_id] = self.goal[3:] - sites_offset[site_id]
        self.sim.forward()

    # Utilities
    # ----------------------------

    def _get_achieved_goal(self):
        if self.has_object:
            # Achieved goal is object position
            raise NotImplementedError
        else:
            # Achieved goal is gripper(s) position(s)
            ag = np.zeros(6)
            if self.has_left_arm:
                ag[:3] = self.sim.data.get_site_xpos('gripper_l_center').copy()
            if self.has_right_arm:
                ag[3:] = self.sim.data.get_site_xpos('gripper_r_center').copy()
            return ag

    @staticmethod
    def get_urdf_model():
        from urdf_parser_py.urdf import URDF
        root_dir = os.path.dirname(__file__)
        model = URDF.from_xml_file(os.path.join(root_dir, 'assets/misc/yumi.urdf'))
        return model

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

    def _set_sim_state(self, qpos, qvel):
        assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


class YumiReachRightArmEnv(YumiEnv, EzPickle):
    def __init__(self, **kwargs):
        super().__init__(arm='right', block_gripper=True, reward_type='sparse',
                         distance_threshold=0.05, has_object=False, **kwargs)
        EzPickle.__init__(self)


class YumiReachLeftArmEnv(YumiEnv, EzPickle):
    def __init__(self, **kwargs):
        super().__init__(arm='left', block_gripper=True, reward_type='sparse',
                         distance_threshold=0.05, has_object=False, **kwargs)
        EzPickle.__init__(self)


class YumiReachTwoArmsEnv(YumiEnv, EzPickle):
    def __init__(self, **kwargs):
        super().__init__(arm='both', block_gripper=True, reward_type='sparse',
                         distance_threshold=0.05, has_object=False, **kwargs)
        EzPickle.__init__(self)
