import os
import copy

import mujoco_py
import numpy as np
from gym.utils import EzPickle
from gym.envs.robotics.robot_env import RobotEnv
from gym.envs.robotics.utils import reset_mocap2body_xpos, reset_mocap_welds


def _check_range(a, a_min, a_max, include_bounds=True):
    if include_bounds:
        return np.all((a_min <= a) & (a <= a_max))
    else:
        return np.all((a_min < a) & (a < a_max))


def _ctrl_set_action(sim, action):
    # Originally from gym.envs.robotics.utils
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def _mocap_set_action(sim, action):
    # Originally from gym.envs.robotics.utils
    if sim.model.nmocap > 0:
        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]
        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


class YumiEnv(RobotEnv):

    def __init__(self, *, arm, block_gripper, reward_type, distance_threshold, has_object, ignore_target_rotation=True):

        if arm not in ['right', 'left', 'both']:
            raise ValueError
        self.arm = arm

        if reward_type not in ['sparse', 'dense']:
            raise ValueError
        self.reward_type = reward_type

        self.block_gripper = block_gripper
        self.has_object = has_object
        self.distance_threshold = distance_threshold
        self.ignore_target_rotation = ignore_target_rotation

        self._table_safe_bounds = (np.r_[-0.20, -0.43], np.r_[0.35, 0.43])
        self._target_bounds_l = (np.r_[-0.20, 0.07, 0.05], np.r_[0.35, 0.43, 0.6])
        self._target_bounds_r = (np.r_[-0.20, -0.43, 0.05], np.r_[0.35, -0.07, 0.6])
        self._obj_target_bounds = (np.r_[-0.15, -0.15, 0.05], np.r_[0.15, 0.15, 0.25])

        self._gripper_r_joint_idx = None
        self._gripper_l_joint_idx = None
        self._arm_r_joint_idx = None
        self._arm_l_joint_idx = None

        self._gripper_joint_max = 0.02
        n_actions = 7
        if not block_gripper:
            n_actions += 1
        if arm == 'both':
            n_actions *= 2

        if self.has_object:
            object_xml = """
            <body name="object0" pos="0.025 0.025 0.025">
                <joint name="object0:joint" type="free" damping="0.01"/>
                <geom size="0.005 0.150 0.025" type="box" condim="4" name="object0" material="block_mat" mass="0.2" friction="1 0.95 0.01" solimp="0.99 0.99 0.01" solref="0.01 1"/>
                <geom size="0.025 0.150 0.005" pos="0 0 -0.02" type="box" condim="4" name="object0_base" material="block_mat" mass="1.8" solimp="0.99 0.99 0.01" solref="0.01 1"/>
                <site name="object0:center" pos="0 0 0" size="0.02 0.02 0.02" rgba="0 0 1 1" type="sphere"/>
                <site name="object0:left" pos="0 0.125 0" size="0.02 0.02 0.02" rgba="0 1 0 1" type="sphere"/>
                <site name="object0:right" pos="0 -0.125 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"/>
            </body>
            
            <body name="target0" pos="1 0.87 0.2">
                <geom size="0.005 0.150 0.025" type="box" name="target0" material="block_mat_target" contype="0" conaffinity="0"/>
                <geom size="0.025 0.150 0.005" pos="0 0 -0.02" type="box" name="target0_base" material="block_mat_target" contype="0" conaffinity="0"/>
                <site name="target0:center" pos="0 0 0" size="0.02 0.02 0.02" rgba="0 0 1 0.5" type="sphere"/>
                <site name="target0:left" pos="0 0.125 0" size="0.02 0.02 0.02" rgba="0 1 0 0.5" type="sphere"/>
                <site name="target0:right" pos="0 -0.125 0" size="0.02 0.02 0.02" rgba="1 0 0 0.5" type="sphere"/>
            </body>
            """
        else:
            object_xml = ""

        model_path = os.path.join(os.path.dirname(__file__), 'assets', f'yumi_{arm}.xml')
        super(YumiEnv, self).__init__(model_path=model_path, n_substeps=5,
                                      n_actions=n_actions, initial_qpos=None, xml_format=dict(object=object_xml))

    def mocap_control(self, action):
        reset_mocap2body_xpos(self.sim)
        self.sim.model.eq_active[:] = 1
        _mocap_set_action(self.sim, action)
        self.sim.step()
        self.sim.model.eq_active[:] = 0

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict):
        assert achieved_goal.shape == desired_goal.shape
        if self.reward_type == 'sparse':
            if self.has_object:
                success = self._is_success(achieved_goal, desired_goal)
                return success - 1
            else:
                success_l = float(self.has_left_arm) * self._is_success(achieved_goal[..., :3], desired_goal[..., :3])
                success_r = float(self.has_right_arm) * self._is_success(achieved_goal[..., 3:], desired_goal[..., 3:])
                success = success_l + success_r
                if self.has_two_arms:
                    return success - 2
                else:
                    return success - 1
        else:
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return -d

    # RobotEnv methods
    # ----------------------------

    def _reset_sim(self):

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        if self._arm_l_joint_idx is not None:
            pos_low_l = np.r_[0.8, -0.3, -0.4, -0.4, -0.3, -0.3, -0.3]
            pos_high_l = np.r_[1.4, 0.6, 0.4, 0.4, 0.3, 0.3, 0.3]
            self.init_qpos[self._arm_l_joint_idx] = self.np_random.uniform(pos_low_l, pos_high_l)
            self.init_qvel[self._arm_l_joint_idx] = 0.0

        if self._gripper_l_joint_idx is not None:
            self.init_qpos[self._gripper_l_joint_idx] = 0.0
            self.init_qvel[self._gripper_l_joint_idx] = 0.0

        if self._arm_r_joint_idx is not None:
            pos_low_r = np.r_[-1.4, -0.3, -0.4, -0.4, -0.3, -0.3, -0.3]
            pos_high_r = np.r_[-0.8,  0.6,  0.4,  0.4,  0.3,  0.3,  0.3]
            self.init_qpos[self._arm_r_joint_idx] = self.np_random.uniform(pos_low_r, pos_high_r)
            self.init_qvel[self._arm_r_joint_idx] = 0.0

        if self._gripper_r_joint_idx is not None:
            self.init_qpos[self._gripper_r_joint_idx] = 0.0
            self.init_qvel[self._gripper_r_joint_idx] = 0.0

        self.sim.data.ctrl[:] = 0.0
        self._set_sim_state(qpos, qvel)

        if self.has_left_arm:
            # make sure the left gripper is above the table
            gripper_z = self.sim.data.get_site_xpos('gripper_l_center')[2]
            if gripper_z < 0.043:
                return False

        if self.has_right_arm:
            # make sure the right gripper is above the table
            gripper_z = self.sim.data.get_site_xpos('gripper_r_center')[2]
            if gripper_z < 0.043:
                return False

        return True

    def _get_obs(self):

        arm_l_qpos = np.zeros(0)
        arm_l_qvel = np.zeros(0)
        gripper_l_qpos = np.zeros(0)
        gripper_l_pos = np.zeros(0)
        gripper_l_vel = np.zeros(0)
        gripper_l_to_obj = np.zeros(0)

        arm_r_qpos = np.zeros(0)
        arm_r_qvel = np.zeros(0)
        gripper_r_qpos = np.zeros(0)
        gripper_r_pos = np.zeros(0)
        gripper_r_vel = np.zeros(0)
        gripper_r_to_obj = np.zeros(0)

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        if self.has_left_arm:
            arm_l_qpos = self.sim.data.qpos[self._arm_l_joint_idx]
            arm_l_qvel = self.sim.data.qvel[self._arm_l_joint_idx]
            arm_l_qvel = np.clip(arm_l_qvel, -10, 10)
            gripper_l_pos = self.sim.data.get_site_xpos('gripper_l_center').copy()
            gripper_l_vel = self.sim.data.get_site_xvelp('gripper_l_center') * dt

        if self._gripper_l_joint_idx is not None:
            gripper_l_qpos = self.sim.data.qpos[self._gripper_l_joint_idx]

        if self.has_right_arm:
            arm_r_qpos = self.sim.data.qpos[self._arm_r_joint_idx]
            arm_r_qvel = self.sim.data.qvel[self._arm_r_joint_idx]
            arm_r_qvel = np.clip(arm_r_qvel, -10, 10)
            gripper_r_pos = self.sim.data.get_site_xpos('gripper_r_center').copy()
            gripper_r_vel = self.sim.data.get_site_xvelp('gripper_r_center') * dt

        if self._gripper_r_joint_idx is not None:
            gripper_r_qpos = self.sim.data.qpos[self._gripper_r_joint_idx]

        object_pose = np.zeros(0)
        object_velp = np.zeros(0)
        object_velr = np.zeros(0)
        if self.has_object:
            # Achieved goal is object position and quaternion
            object_pose = np.zeros(7)
            object_pose[:3] = self.sim.data.get_body_xpos('object0')
            if not self.ignore_target_rotation:
                object_pose[3:] = self.sim.data.get_body_xquat('object0')
            achieved_goal = object_pose.copy()
            if self.has_left_arm:
                gripper_l_to_obj = self.sim.data.get_site_xpos('object0:left') - gripper_l_pos
            if self.has_right_arm:
                gripper_r_to_obj = self.sim.data.get_site_xpos('object0:right') - gripper_r_pos
            object_velp = self.sim.data.get_site_xvelp('object0:center') * dt
            object_velr = self.sim.data.get_site_xvelr('object0:center') * dt
        else:
            # Achieved goal is gripper(s) position(s)
            achieved_goal = np.zeros(6)
            if self.has_left_arm:
                achieved_goal[:3] = gripper_l_pos.copy()
            if self.has_right_arm:
                achieved_goal[3:] = gripper_r_pos.copy()

        obs = np.concatenate([
            arm_l_qpos, arm_l_qvel, gripper_l_qpos,
            arm_r_qpos, arm_r_qvel, gripper_r_qpos,
            gripper_l_pos, gripper_r_pos,
            gripper_l_vel, gripper_r_vel,
            gripper_l_to_obj, gripper_r_to_obj,
            object_pose, object_velp, object_velr,
        ])

        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal.copy(),
        }

    def _set_action(self, a):
        a = np.clip(a, self.action_space.low, self.action_space.high)

        if not self.block_gripper:
            arm1_a = a[:8]
            arm2_a = a[8:]
            # remap [-1, 1] to [0, gripper_joint_max]
            gripper1_a = self._gripper_joint_max * (arm1_a[7:] + 1.0) / 2.0
            gripper2_a = self._gripper_joint_max * (arm2_a[7:] + 1.0) / 2.0
            a = np.r_[arm1_a, gripper1_a, arm2_a, gripper2_a]
        else:
            arm1_a = a[:7]
            arm2_a = a[7:]
            g = self._gripper_joint_max
            a = np.r_[arm1_a, g, g]
            if self.has_two_arms:
                a = np.r_[a, arm2_a, g, g]

        _ctrl_set_action(self.sim, a)
        return a

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        if self.has_object:
            # Goal is object target position and quaternion
            new_goal = np.zeros(7)
            new_goal[:3] = self.np_random.uniform(*self._obj_target_bounds)
            # TODO: Randomize rotation
        else:
            # Goal is gripper(s) target position(s)
            new_goal = np.zeros(6)
            old_state = copy.deepcopy(self.sim.get_state())
            if self.has_left_arm:
                while True:
                    left_arm_q = self._sample_safe_qpos(self._arm_l_joint_idx)
                    grp_l_pos = self._fk_position(left_arm_q=left_arm_q, restore_state=False)
                    if _check_range(grp_l_pos, *self._target_bounds_l):
                        new_goal[:3] = grp_l_pos
                        break
            if self.has_right_arm:
                while True:
                    right_arm_q = self._sample_safe_qpos(self._arm_r_joint_idx)
                    grp_r_pos = self._fk_position(right_arm_q=right_arm_q, restore_state=False)
                    if _check_range(grp_r_pos, *self._target_bounds_r):
                        new_goal[3:] = grp_r_pos
                        break
            self.sim.set_state(old_state)
            self.sim.forward()
        return new_goal

    def _env_setup(self, initial_qpos):
        if initial_qpos is not None:
            raise NotImplementedError
        reset_mocap_welds(self.sim)
        self.sim.forward()

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        yumi_arm_joints = [1, 2, 7, 3, 4, 5, 6]
        if self.has_right_arm:
            self._arm_r_joint_idx = [self.sim.model.joint_name2id(f'yumi_joint_{i}_r') for i in yumi_arm_joints]
            self.arm_r_joint_lims = self.sim.model.jnt_range[self._arm_r_joint_idx].copy()
        if self.has_left_arm:
            self._arm_l_joint_idx = [self.sim.model.joint_name2id(f'yumi_joint_{i}_l') for i in yumi_arm_joints]
            self.arm_l_joint_lims = self.sim.model.jnt_range[self._arm_l_joint_idx].copy()

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

        self._reset_sim()

    def _viewer_setup(self):
        self.viewer.cam.distance = 1.7
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 180

    def _render_callback(self):
        # Visualize target.
        if self.has_object:
            bodies_offset = (self.sim.data.body_xpos - self.sim.model.body_pos).copy()
            body_id = self.sim.model.body_name2id('target0')
            self.sim.model.body_pos[body_id, :] = self.goal[:3] - bodies_offset[body_id]
        else:
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

    def _sample_safe_qpos(self, arm_joint_idx):
        margin = np.pi/6
        jnt_range = self.sim.model.jnt_range[arm_joint_idx].copy()
        jnt_range[:, 0] += margin
        jnt_range[:, 1] -= margin
        return self.np_random.uniform(*jnt_range.T)

    def _fk_position(self, left_arm_q=None, right_arm_q=None, restore_state=True):
        grp_pos, old_state = None, None
        if restore_state:
            old_state = copy.deepcopy(self.sim.get_state())
        if left_arm_q is not None:
            assert right_arm_q is None
            idx = self.sim.model.jnt_qposadr[self._arm_l_joint_idx]
            self.sim.data.qpos[idx] = left_arm_q
            self.sim.forward()
            grp_pos = self.sim.data.get_site_xpos('gripper_l_center').copy()
        if right_arm_q is not None:
            assert left_arm_q is None
            idx = self.sim.model.jnt_qposadr[self._arm_r_joint_idx]
            self.sim.data.qpos[idx] = right_arm_q
            self.sim.forward()
            grp_pos = self.sim.data.get_site_xpos('gripper_r_center').copy()
        if restore_state:
            self.sim.set_state(old_state)
            self.sim.forward()
        return grp_pos


class YumiReachEnv(YumiEnv, EzPickle):
    def __init__(self, **kwargs):
        default_kwargs = dict(block_gripper=True, reward_type='sparse', distance_threshold=0.05)
        merged = {**default_kwargs, **kwargs}
        super().__init__(has_object=False, **merged)
        EzPickle.__init__(self)


class YumiReachRightArmEnv(YumiReachEnv):
    def __init__(self, **kwargs):
        super().__init__(arm='right', **kwargs)


class YumiReachLeftArmEnv(YumiReachEnv):
    def __init__(self, **kwargs):
        super().__init__(arm='left', **kwargs)


class YumiReachTwoArmsEnv(YumiReachEnv):
    def __init__(self, **kwargs):
        super().__init__(arm='both', **kwargs)


class YumiBarEnv(YumiEnv, EzPickle):
    def __init__(self, **kwargs):
        default_kwargs = dict(arm='both', block_gripper=False, reward_type='sparse', distance_threshold=0.05)
        merged = {**default_kwargs, **kwargs}
        super().__init__(has_object=True, **merged)
        EzPickle.__init__(self)
