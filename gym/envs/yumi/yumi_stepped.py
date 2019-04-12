import copy

import numpy as np
import gym
from gym import spaces
from gym.envs.yumi import YumiLiftEnv
from gym.envs.robotics.utils import reset_mocap2body_xpos
from gym.utils import transformations as tf


class YumiSteppedEnv(gym.Env):

    def __init__(self, *, render_substeps=False):
        super(YumiSteppedEnv, self).__init__()

        self.metadata = {
            'render.modes': ['human'],
        }

        self._qp_solver = None
        self.render_substeps = render_substeps
        self.sim_env = YumiLiftEnv()
        obs = self._get_obs()

        n_actions = (3 + 1) * 2 # pos of grippers + conf of grippers
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

    # Shortcuts
    # ----------------------------

    @property
    def viewer(self):
        return self.sim_env.viewer

    @property
    def sim(self):
        return self.sim_env.sim

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

    def get_gripper_pose(self, arm):
        assert arm in ('l', 'r')
        return np.r_[
            self.sim.data.get_site_xpos(f'gripper_{arm}_center'),
            tf.rotations.mat2quat(self.sim.data.get_site_xmat(f'gripper_{arm}_center')),
        ]

    def get_gripper_pos(self, arm):
        assert arm in ('l', 'r')
        return self.sim.data.get_site_xpos(f'gripper_{arm}_center').copy()

    def get_gripper_vel(self, arm):
        assert arm in ('l', 'r')
        return self.sim.data.get_site_xvelp(f'gripper_{arm}_center') * self.sim_env.dt

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

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.sim_env.seed(seed)

    def step(self, action: np.ndarray):

        def failure():
            return self._get_obs(), 0.0, True, dict()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.reshape(-1, 4)
        grippers_pos_wrt_obj = action[:, :3] * self.get_task_space()
        grippers_conf = action[:, 3]

        obj_pose = self.get_object_pose()
        grippers_pos_targets = np.array([
            tf.apply_tf(np.r_[transf, 1., 0., 0., 0.], obj_pose)[:3] for transf in grippers_pos_wrt_obj
        ])

        vec = grippers_pos_targets[0, :2] - grippers_pos_targets[1, :2]
        grasp_radius = np.linalg.norm(vec, ord=2) / 2.0
        right_yaw = np.arctan2(vec[1], vec[0]) + np.pi/2
        left_yaw = np.arctan2(-vec[1], -vec[0]) + np.pi/2

        left_target_pose = np.r_[
            grippers_pos_targets[0],
            tf.rotations.euler2quat(np.r_[0, 0, left_yaw])
        ]

        right_target_pose = np.r_[
            grippers_pos_targets[1],
            tf.rotations.euler2quat(np.r_[0, 0, right_yaw])
        ]

        # - move arms to pregrasp1 pose
        # targets are farther and higher wrt object center
        max_pos_err = self._move_arms(
            left_target=tf.apply_tf(np.r_[0., 0.05, 0.1], left_target_pose)[:3], left_yaw=left_yaw,
            right_target=tf.apply_tf(np.r_[0., 0.05, 0.1], right_target_pose)[:3], right_yaw=right_yaw,
            left_grp_config=grippers_conf[0], right_grp_config=grippers_conf[1], max_steps=120,
        )

        if max_pos_err > 0.05 or self.is_object_unreachable():
            return failure()

        # - move arms to pregrasp2 pose
        # targets are farther and but aligned to object center
        max_pos_err = self._move_arms(
            left_target=tf.apply_tf(np.r_[0., 0.05, 0.01], left_target_pose)[:3], left_yaw=left_yaw,
            right_target=tf.apply_tf(np.r_[0., 0.05, 0.01], right_target_pose)[:3], right_yaw=right_yaw,
            left_grp_config=grippers_conf[0], right_grp_config=grippers_conf[1], max_steps=50,
        )

        if max_pos_err > 0.05 or self.is_object_unreachable():
            return failure()

        # - move arms to grasp pose
        # targets are as specified by the agent
        self._move_arms(
            left_target=tf.apply_tf(np.r_[0., 0., 0.01], left_target_pose)[:3], left_yaw=left_yaw,
            right_target=tf.apply_tf(np.r_[0., 0., 0.01], right_target_pose)[:3], right_yaw=right_yaw,
            left_grp_config=grippers_conf[0], right_grp_config=grippers_conf[1], max_steps=40,
        )

        if self.is_object_unreachable():
            return failure()

        # - lift
        # targets are as specified by the agent but a bit higher to go up
        stable_steps = self._move_arms(
            left_target=np.r_[0., grasp_radius * 0.9, 0.05], left_yaw=left_yaw,
            right_target=np.r_[0., -grasp_radius * 0.9, 0.05], right_yaw=right_yaw,
            count_stable_steps=True, targets_relative_to=self.get_object_pos,
            left_grp_config=grippers_conf[0], right_grp_config=grippers_conf[1], max_steps=200,
        )

        obs = self._get_obs()
        reward = float(stable_steps)
        done = self.is_object_unreachable()
        info = dict()
        return obs, reward, done, info

    def render(self, mode='human', keep_markers=False):
        markers = []
        if keep_markers:
            markers = copy.deepcopy(self.viewer._markers)
        self.sim_env.render()
        for m in markers:
            self.viewer.add_marker(**m)

    def reset(self):
        self.sim.model.eq_active[:] = 0
        self.sim_env.reset()
        reset_mocap2body_xpos(self.sim)
        self.sim.model.eq_active[:] = 1
        self.sim.step()
        return self._get_obs()

    # GoalEnv methods
    # ----------------------------

    def _get_obs(self):
        obj_pos = self.get_object_pos()
        grippers_pos = np.r_[self.get_gripper_pos('l'), self.get_gripper_pos('r')]
        obs = np.r_[obj_pos, grippers_pos]
        return obs

    # Arm control
    # ----------------------------

    def _controller(self, error, prev_error, k):
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

            for arm_i, arm in enumerate(('l', 'r')):

                curr_pose = self.get_gripper_pose(arm)
                curr_q = self.get_arm_config(arm)

                if arm == 'l':
                    pitch = np.pi - 0.9
                    u_masked = u[:7]
                    prev_err = prev_err_l
                else:
                    pitch = np.pi - 0.9
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

                if self.render_substeps:
                    tf.render_pose(target_pos.copy(), self.viewer, label=f"{arm}_p", unique_label=True)
                    tf.render_pose(target_pose.copy(), self.viewer, label=f"{arm}_t", unique_label=True)
                    tf.render_pose(curr_pose.copy(), self.viewer, label=f"{arm}", unique_label=True)

            grasp_center_pos /= 2.0

            u[7] = left_grp_config
            u[15] = right_grp_config
            u = np.clip(u, self.sim_env.action_space.low, self.sim_env.action_space.high)
            self.sim_env.step(u)

            if self.render_substeps:
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
