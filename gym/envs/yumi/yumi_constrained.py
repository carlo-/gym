import copy

import numpy as np
import gym
from gym import spaces
from gym.envs.yumi.yumi_env import YumiEnv, YumiTask
from gym.utils import transformations as tf
from gym.envs.robotics.utils import reset_mocap2body_xpos


def _goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    assert goal_a.shape[-1] == 3
    delta_pos = goal_a[..., :3] - goal_b[..., :3]
    d_pos = np.linalg.norm(delta_pos, axis=-1)
    return d_pos


def _mocap_set_action(sim, action):
    # Originally from gym.envs.robotics.utils
    if sim.model.nmocap > 0:
        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]
        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] += pos_delta
        sim.data.mocap_quat[:] += quat_delta


class YumiConstrainedEnv(gym.GoalEnv):

    def __init__(self, *, reward_type='dense', rotation_ctrl=False, fingers_ctrl=False, distance_threshold=0.05,
                 randomize_initial_object_pos=True, mocap_ctrl=False, render_poses=True,
                 object_id='fetch_box', **kwargs):
        super(YumiConstrainedEnv, self).__init__()

        self.metadata = {
            'render.modes': ['human'],
        }

        self.sim_env = YumiEnv(
            arm='both', block_gripper=False, reward_type=reward_type, task=YumiTask.PICK_AND_PLACE_OBJECT,
            object_id=object_id, randomize_initial_object_pos=randomize_initial_object_pos, **kwargs,
        )
        self.mocap_ctrl = mocap_ctrl
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.render_poses = render_poses
        obs = self._get_obs()

        n_actions = 4 # dist between grippers (1) + grasp center pos delta (3)
        if rotation_ctrl:
            n_actions += 4 # grasp center quat delta (4)
        if fingers_ctrl:
            n_actions += 2 # # dist between fingers, (left, right) grippers (2)
        self.rotation_ctrl_enabled = rotation_ctrl
        self.fingers_ctrl_enabled = fingers_ctrl

        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    # Shortcuts
    # ----------------------------

    def sync_object_init_pos(self, pos: np.ndarray, wrt_table=False, now=False):
        self.sim_env.sync_object_init_pos(pos, wrt_table=wrt_table, now=now)

    @property
    def viewer(self):
        return self.sim_env.viewer

    @property
    def sim(self):
        return self.sim_env.sim

    @property
    def np_random(self):
        return self.sim_env.np_random

    def _get_object_size(self):
        geom = self.sim.model.geom_name2id('object0_base')
        return self.sim.model.geom_size[geom].copy()

    def get_object_pose(self):
        return np.r_[
            self.sim.data.get_body_xpos('object0'),
            self.sim.data.get_body_xquat('object0'),
        ]

    def get_object_pos(self):
        return self.sim.data.get_body_xpos('object0').copy()

    def get_object_rot(self):
        return tf.rotations.mat2euler(self.sim.data.get_site_xmat('object0:center'))

    def get_object_velp(self):
        return self.sim.data.get_site_xvelp('object0:center') * self.sim_env.dt

    def get_object_velr(self):
        return self.sim.data.get_site_xvelr('object0:center') * self.sim_env.dt

    def get_gripper_pose(self, arm):
        assert arm in ('l', 'r')
        return np.r_[
            self.sim.data.get_site_xpos(f'gripper_{arm}_center'),
            tf.rotations.mat2quat(self.sim.data.get_site_xmat(f'gripper_{arm}_center')),
        ]

    def get_gripper_pos(self, arm):
        assert arm in ('l', 'r')
        return self.sim.data.get_site_xpos(f'gripper_{arm}_center').copy()

    def get_gripper_velp(self, arm):
        assert arm in ('l', 'r')
        return self.sim.data.get_site_xvelp(f'gripper_{arm}_center') * self.sim_env.dt

    def get_grasp_center_pos(self):
        return (self.get_gripper_pos('l') + self.get_gripper_pos('r')) / 2.0

    def get_grasp_center_velp(self):
        return (self.get_gripper_velp('l') + self.get_gripper_velp('r')) / 2.0

    def get_arm_config(self, arm):
        if arm == 'l':
            idx = self.sim_env._arm_l_joint_idx
        else:
            idx = self.sim_env._arm_r_joint_idx
        return self.sim.data.qpos[idx].copy()

    def get_task_space(self):
        obj_radius = self._get_object_size()[0]
        return np.r_[obj_radius*1.0, obj_radius*1.0, 0.03]

    def is_object_on_ground(self):
        obj_pose = self.get_object_pos()
        return obj_pose[2] < 0.0

    def is_object_unreachable(self):
        obj_pose = self.get_object_pos()
        return not (
            (-0.15 < obj_pose[0] < 0.15)
            and (-0.15 < obj_pose[1] < 0.15)
            and (-0.01 < obj_pose[2] < 0.45)
        )

    def get_table_surface_pose(self):
        return self.sim_env.get_table_surface_pose()

    def _unpack_action(self, action):
        dist_between_grippers = action[0]
        grasp_center_pos_delta = action[1:4]
        offset = 4

        if self.rotation_ctrl_enabled:
            grasp_center_quat_delta = action[offset:offset+4]
            offset += 4
        else:
            grasp_center_quat_delta = np.r_[1., 0., 0., 0.]

        if self.fingers_ctrl_enabled:
            l_fingers_ctrl, r_fingers_ctrl = action[offset:offset+2]
        else:
            l_fingers_ctrl = r_fingers_ctrl = -1.

        return (
            dist_between_grippers, grasp_center_pos_delta,
            grasp_center_quat_delta, l_fingers_ctrl, r_fingers_ctrl
        )

    def _reset(self):

        qpos = np.r_[
            0.5331, 0.046, 0.3148, -0.0206, -0.8999, 1.0898, -2.1024, 0.0, 0.0002, -0.0875, 0.0288, -0.6761,
            0.0196, 0.841, 1.0012, -0.9585, -0.0, 0.0001, 0.025, 0.025, 0.03, 1.0, -0.0, 0.0, 0.0
        ]

        if self.sim_env.has_rotating_platform:
            qpos = np.r_[-0.75, qpos]

        qvel = np.zeros_like(self.sim_env.init_qvel)

        self.sim.data.ctrl[:] = 0.0
        self.sim_env._set_sim_state(qpos, qvel)

        object_qpos = self.sim.data.get_joint_qpos('object0:joint').copy()
        self.sim_env._object_xy_pos_to_sync = self.np_random.uniform(*self.sim_env._obj_init_bounds)
        if self.sim_env.has_rotating_platform:
            object_qpos[2] += 0.020
            object_qpos[:2] = self.sim.data.get_site_xpos('rotating_platform:far_end')[:2]
        elif self.sim_env.has_button:
            object_qpos[:2] = 0.375, -0.476
        elif self.sim_env.randomize_initial_object_pos:
            # Randomize initial position of object.
            object_qpos[:2] = self.sim_env._object_xy_pos_to_sync.copy()
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim_env._reset_button()

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.sim_env.seed(seed)

    def step(self, action: np.ndarray):

        action = np.clip(action, self.action_space.low, self.action_space.high)

        (dist_between_grippers, grasp_center_pos_delta,
         grasp_center_quat_delta, l_fingers_ctrl, r_fingers_ctrl) = self._unpack_action(action)

        if self.mocap_ctrl:
            pos_k = 0.1
            dist_range = [0.05, 0.20]
        else:
            pos_k = 0.2
            dist_range = [0.05, 0.30]

        dist_between_grippers = np.interp(dist_between_grippers, [-1, 1], dist_range)

        curr_grasp_center_pose = self.get_grasp_center_pos() # TODO: Use pose rather than position only
        target_grasp_center_pose = np.r_[0., 0., 0., 1., 0., 0., 0.]

        target_grasp_center_pose[:3] = curr_grasp_center_pose[:3] + grasp_center_pos_delta * pos_k
        if self.rotation_ctrl_enabled:
            # target_grasp_center_pose[3:] = curr_grasp_center_pose[3:] + grasp_center_quat_delta * 0.1
            raise NotImplementedError

        grippers_pos_targets = np.array([
            tf.apply_tf(np.r_[0., dist_between_grippers/2.0, 0., 1., 0., 0., 0.], target_grasp_center_pose)[:3],
            tf.apply_tf(np.r_[0., -dist_between_grippers/2.0, 0., 1., 0., 0., 0.], target_grasp_center_pose)[:3]
        ])

        vec = grippers_pos_targets[0, :2] - grippers_pos_targets[1, :2]
        right_yaw = np.arctan2(vec[1], vec[0]) + np.pi/2
        left_yaw = np.arctan2(-vec[1], -vec[0]) + np.pi/2

        grasp_radius = np.linalg.norm(vec, ord=2) / 2.0
        assert np.allclose(grasp_radius, dist_between_grippers / 2.0)

        if self.mocap_ctrl:
            self._move_arms_mocap(
                left_target=grippers_pos_targets[0], left_yaw=left_yaw,
                right_target=grippers_pos_targets[1], right_yaw=right_yaw,
                left_grp_config=l_fingers_ctrl, right_grp_config=r_fingers_ctrl, max_steps=1,
            )
        else:
            self._move_arms(
                left_target=grippers_pos_targets[0], left_yaw=left_yaw,
                right_target=grippers_pos_targets[1], right_yaw=right_yaw,
                left_grp_config=l_fingers_ctrl, right_grp_config=r_fingers_ctrl, max_steps=5,
            )

        obs = self._get_obs()
        done = False # self.is_object_unreachable()
        info = {'is_success': self._is_success(obs['achieved_goal'], self.goal)}
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def render(self, mode='human', keep_markers=False):
        markers = []
        if keep_markers:
            markers = copy.deepcopy(self.viewer._markers)
        self.sim_env.render()
        for m in markers:
            self.viewer.add_marker(**m)

    def reset(self):
        self._reset()
        self.sim_env.goal = self._sample_goal().copy()
        self.sim.step()
        return self._get_obs()

    # GoalEnv methods
    # ----------------------------

    @property
    def goal(self):
        return self.sim_env.goal[:3]

    def _sample_goal(self):
        new_goal = self.sim_env._sample_goal()
        return new_goal

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, info: dict):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, goal).astype(np.float32)
            return success - 1.
        else:
            d = _goal_distance(achieved_goal, goal)
            return -d

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        d = _goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _get_obs(self):

        if self.sim_env.is_pressing_button():
            self.sim_env._did_press_button()

        grippers_pos = np.r_[self.get_gripper_pos('l'), self.get_gripper_pos('r')]
        grippers_velp = np.r_[self.get_gripper_velp('l'), self.get_gripper_velp('r')]

        grasp_center_pos = self.get_grasp_center_pos()
        grasp_center_velp = self.get_grasp_center_velp()
        object_pos = self.get_object_pos()
        object_rot = self.get_object_rot()
        object_velp = self.get_object_velp()
        object_velr = self.get_object_velr()

        object_rel_pos = object_pos - grasp_center_pos
        object_velp -= grasp_center_velp

        achieved_goal = object_pos.copy()

        obs = np.r_[
            grasp_center_pos, grasp_center_velp, grippers_pos, grippers_velp,
            object_pos, object_rot, object_rel_pos, object_velp, object_velr
        ]

        return dict(
            observation=obs,
            achieved_goal=achieved_goal,
            desired_goal=self.goal.copy(),
        )

    # Arm control
    # ----------------------------

    def _controller(self, error, prev_error, k):
        if np.all(prev_error == 0.0):
            d_err = 0.0
        else:
            d_err = (error - prev_error) / self.sim_env.dt
        prev_error[:] = error
        return -(1.0 * error + 0.05 * d_err) * k

    def _move_arms(self, *, left_target: np.ndarray, right_target: np.ndarray, left_yaw=0.0, right_yaw=0.0,
                   pos_threshold=0.02, rot_threshold=0.1, k=2.0, max_steps=100, count_stable_steps=False,
                   targets_relative_to=None, left_grp_config=-1.0, right_grp_config=-1.0):

        targets = {'l': left_target, 'r': right_target}
        yaws = {'l': left_yaw, 'r': right_yaw}
        stable_steps = 0
        prev_rel_pos = np.zeros(3)
        u = np.zeros(self.sim_env.action_space.shape)
        prev_err_l = np.zeros(7)
        prev_err_r = np.zeros(7)
        max_pos_err = -np.inf

        for i in range(max_steps):

            grasp_center_pos = np.zeros(3)
            max_rot_err = -np.inf
            max_pos_err = -np.inf

            d_above_table = self.get_object_pos()[2] - self.sim_env._object_z_offset
            grp_xrot = 0.9 + d_above_table * 2.0

            for arm_i, arm in enumerate(('l', 'r')):

                curr_pose = self.get_gripper_pose(arm)
                curr_q = self.get_arm_config(arm)

                if arm == 'l':
                    pitch = np.pi - grp_xrot
                    u_masked = u[:7]
                    prev_err = prev_err_l
                else:
                    pitch = np.pi - grp_xrot
                    u_masked = u[8:15]
                    prev_err = prev_err_r

                if callable(targets_relative_to):
                    reference = targets_relative_to()
                    target_pos = tf.apply_tf(targets[arm], reference)[:3]
                else:
                    target_pos = targets[arm]

                target_pose = np.r_[target_pos, tf.rotations.euler2quat(np.r_[0., 0., yaws[arm]])]
                target_pose = tf.apply_tf(np.r_[0., 0., 0., tf.rotations.euler2quat(np.r_[pitch, 0., 0.])], target_pose)

                grasp_center_pos += curr_pose[:3]
                max_pos_err = max(max_pos_err, np.abs(curr_pose[:3] - target_pose[:3]).max())
                max_rot_err = max(max_rot_err, tf.quat_angle_diff(curr_pose[3:], target_pose[3:]))

                target_q = self.sim_env.mocap_ik(target_pose - curr_pose, arm)
                u_masked[:] = self._controller(curr_q - target_q, prev_err, k)

                if self.viewer is not None and self.render_poses:
                    tf.render_pose(target_pos.copy(), self.viewer, label=f"{arm}_p", unique_label=True)
                    tf.render_pose(target_pose.copy(), self.viewer, label=f"{arm}_t", unique_label=True)
                    tf.render_pose(curr_pose.copy(), self.viewer, label=f"{arm}", unique_label=True)

            grasp_center_pos /= 2.0

            u[7] = left_grp_config
            u[15] = right_grp_config
            u = np.clip(u, self.sim_env.action_space.low, self.sim_env.action_space.high)

            self.sim_env._set_action(u)
            self.sim.step()
            self.sim_env._step_callback()

            if self.viewer is not None:
                if self.render_poses:
                    tf.render_pose(grasp_center_pos, self.viewer, label="grasp_center", unique_id=5554)
                self.render(keep_markers=True)

            if max_pos_err < pos_threshold and max_rot_err < rot_threshold:
                break

            if count_stable_steps:
                obj_pos = self.get_object_pos()
                rel_pos = obj_pos - grasp_center_pos
                still = prev_rel_pos is not None and np.all(np.abs(rel_pos - prev_rel_pos) < 0.002)
                obj_above_table = len(self.sim_env.get_object_contact_points(other_body='table')) == 0
                if still and obj_above_table:
                    stable_steps += 1
                elif i > 10:
                    break
                prev_rel_pos = rel_pos

        if count_stable_steps:
            return stable_steps

        return max_pos_err

    def _move_arms_mocap(self, *, left_target: np.ndarray, right_target: np.ndarray, left_yaw=0.0, right_yaw=0.0,
                         pos_threshold=0.02, rot_threshold=0.1, k=1.0, max_steps=1,
                         left_grp_config=-1.0, right_grp_config=-1.0):

        targets = {'l': left_target, 'r': right_target}
        yaws = {'l': left_yaw, 'r': right_yaw}
        self.sim.model.eq_active[:] = 1

        for i in range(max_steps):

            grasp_center_pos = np.zeros(3)
            max_rot_err = -np.inf
            max_pos_err = -np.inf

            d_above_table = self.get_object_pos()[2] - self.sim_env._object_z_offset
            grp_xrot = 0.9 + d_above_table * 2.0

            mocap_a = np.zeros((self.sim.model.nmocap, 7))

            for arm_i, arm in enumerate(('l', 'r')):

                curr_pose = self.get_gripper_pose(arm)
                pitch = np.pi - grp_xrot

                target_pos = targets[arm]
                target_pose = np.r_[target_pos, tf.rotations.euler2quat(np.r_[0., 0., yaws[arm]])]
                target_pose = tf.apply_tf(np.r_[0., 0., 0., tf.rotations.euler2quat(np.r_[pitch, 0., 0.])], target_pose)

                grasp_center_pos += curr_pose[:3]
                if max_steps > 1:
                    max_pos_err = max(max_pos_err, np.abs(curr_pose[:3] - target_pose[:3]).max())
                    max_rot_err = max(max_rot_err, tf.quat_angle_diff(curr_pose[3:], target_pose[3:]))

                mocap_a[arm_i] = target_pose - curr_pose

                if self.viewer is not None and self.render_poses:
                    tf.render_pose(target_pos.copy(), self.viewer, label=f"{arm}_p", unique_label=True)
                    tf.render_pose(target_pose.copy(), self.viewer, label=f"{arm}_t", unique_label=True)
                    tf.render_pose(curr_pose.copy(), self.viewer, label=f"{arm}", unique_label=True)

            grasp_center_pos /= 2.0

            # set config of grippers
            u = np.zeros(self.sim_env.action_space.shape)
            u[7] = left_grp_config
            u[15] = right_grp_config
            self.sim_env._set_action(u)

            # set pose of grippers
            _mocap_set_action(self.sim, mocap_a * k)

            # take simulation step
            self.sim.step()
            self.sim_env._step_callback()

            if self.viewer is not None:
                if self.render_poses:
                    tf.render_pose(grasp_center_pos, self.viewer, label="grasp_center", unique_id=5554)
                self.render(keep_markers=True)

            if max_pos_err < pos_threshold and max_rot_err < rot_threshold:
                break
