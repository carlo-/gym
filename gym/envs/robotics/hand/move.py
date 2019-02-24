import os
import numpy as np

from gym import utils, error
from gym.envs.robotics import rotations, hand_env
from gym.envs.robotics.utils import robot_get_obs, reset_mocap_welds, reset_mocap2body_xpos

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, "
                                       "and also perform the setup instructions here: "
                                       "https://github.com/openai/mujoco-py/.)".format(e))

HAND_PICK_AND_PLACE_XML = os.path.join('hand', 'pick_and_place.xml')
HAND_MOVE_AND_REACH_XML = os.path.join('hand', 'move_and_reach.xml')


def _goal_distance(goal_a, goal_b, ignore_target_rotation):
    assert goal_a.shape == goal_b.shape
    assert goal_a.shape[-1] == 7

    delta_pos = goal_a[..., :3] - goal_b[..., :3]
    d_pos = np.linalg.norm(delta_pos, axis=-1)
    d_rot = np.zeros_like(goal_b[..., 0])

    if not ignore_target_rotation:
        quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]
        # Subtract quaternions and extract angle between them.
        quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
        d_rot = angle_diff
    assert d_pos.shape == d_rot.shape
    return d_pos, d_rot


class MovingHandEnv(hand_env.HandEnv, utils.EzPickle):
    def __init__(self, model_path, reward_type, initial_qpos=None, relative_control=False, has_object=False,
                 randomize_initial_position=True, randomize_initial_rotation=True, ignore_rotation_ctrl=False,
                 distance_threshold=0.05, rotation_threshold=0.1, n_substeps=20, ignore_target_rotation=False,
                 success_on_grasp_only=False):

        self.object_range = 0.15
        self.target_range = 0.15
        self.target_in_the_air = True
        self.has_object = has_object
        self.ignore_target_rotation = ignore_target_rotation
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.ignore_rotation_ctrl = ignore_rotation_ctrl
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.reward_type = reward_type
        self.success_on_grasp_only = success_on_grasp_only
        self.forearm_bounds = (np.r_[0.7, 0.3, 0.3], np.r_[1.75, 1.2, 1.1])

        if ignore_rotation_ctrl and not ignore_target_rotation:
            raise ValueError('Target rotation must be ignored if arm cannot rotate! Set ignore_target_rotation=True')

        if success_on_grasp_only:
            if reward_type != 'sparse':
                raise ValueError('Parameter success_on_grasp_only requires sparse rewards!')
            if not has_object:
                raise ValueError('Parameter success_on_grasp_only requires object to be grasped!')

        default_qpos = dict()
        if self.has_object:
            default_qpos['object:joint'] = [1.25, 0.53, 0.4, 1., 0., 0., 0.]
        initial_qpos = initial_qpos or default_qpos

        hand_env.HandEnv.__init__(self, model_path, n_substeps=n_substeps, initial_qpos=initial_qpos,
                                  relative_control=relative_control, arm_control=True)
        utils.EzPickle.__init__(self)

    def _get_body_pose(self, body_name):
        return np.r_[
            self.sim.data.get_body_xpos(body_name),
            self.sim.data.get_body_xquat(body_name)
        ]

    def _get_achieved_goal(self):
        palm_pose = self._get_body_pose('robot0:ffknuckle')

        if self.has_object:
            pose = self._get_body_pose('object')
        else:
            pose = palm_pose

        if self.ignore_target_rotation:
            pose[3:] = 0.0

        if self.success_on_grasp_only:
            d = np.linalg.norm(palm_pose[:3] - pose[:3])
            return np.r_[pose, d]

        return pose

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, info: dict):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, goal).astype(np.float32)
            return success - 1.
        else:
            d_pos, d_rot = _goal_distance(achieved_goal, goal, self.ignore_target_rotation)
            # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
            # dominated by `d_rot` (in radians).
            return -(10. * d_pos + d_rot)

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):

        assert action.shape == self.action_space.shape
        hand_ctrl = action[:20]
        forearm_ctrl = action[20:] * 0.1

        # set hand action
        hand_env.HandEnv._set_action(self, hand_ctrl)

        # set forearm action
        assert self.sim.model.nmocap == 1
        pos_delta = forearm_ctrl[:3]
        quat_delta = forearm_ctrl[3:]

        if self.ignore_rotation_ctrl:
            quat_delta *= 0.0

        new_pos = self.sim.data.mocap_pos[0] + pos_delta
        new_quat = self.sim.data.mocap_quat[0] + quat_delta

        reset_mocap2body_xpos(self.sim)
        self.sim.data.mocap_pos[0, :] = np.clip(new_pos, *self.forearm_bounds)
        self.sim.data.mocap_quat[0, :] = new_quat

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        d_pos, d_rot = _goal_distance(achieved_goal[..., :7], desired_goal[..., :7], self.ignore_target_rotation)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        achieved_all = achieved_pos * achieved_rot
        if self.success_on_grasp_only:
            assert achieved_goal.shape[-1] == 8
            d_palm = achieved_goal[..., 7]
            achieved_grasp = (d_palm < 0.08).astype(np.float32)
            achieved_all *= achieved_grasp
        return achieved_all

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        forearm_pos = np.array([1.25, 0.75, 0.8])
        forearm_quat = rotations.euler2quat(np.r_[0., 1.97, 1.57])
        self.sim.data.set_mocap_pos('robot0:mocap', forearm_pos)
        self.sim.data.set_mocap_quat('robot0:mocap', forearm_quat)
        for _ in range(10):
            self.sim.step()

        self.initial_arm_xpos = self.sim.data.get_body_xpos('robot0:forearm').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object:center')[2]

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:forearm')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize initial position of object.
        if self.has_object:
            object_qpos = self.sim.data.get_joint_qpos('object:joint').copy()

            object_xpos = self.initial_arm_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_arm_xpos[:2]) < 0.1:
                offset = self.np_random.uniform(-self.object_range, self.object_range, size=2)
                object_xpos = self.initial_arm_xpos[:2] + offset

            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.initial_arm_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        if self.has_object:
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        goal = np.r_[goal, np.zeros(4)]
        if self.success_on_grasp_only:
            goal = np.r_[goal, 0.]
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()[:7]
        assert goal.shape == (7,)
        self.sim.data.set_joint_qpos('target:joint', goal)
        self.sim.data.set_joint_qvel('target:joint', np.zeros(6))

        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.
        self.sim.forward()

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)

        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        forearm_pos = self.sim.data.get_body_xpos('robot0:forearm')
        forearm_rot = rotations.mat2euler(self.sim.data.get_body_xmat('robot0:forearm'))
        forearm_velp = self.sim.data.get_body_xvelp('robot0:forearm') * dt

        object_pose = np.zeros(0)
        object_vel = np.zeros(0)
        if self.has_object:
            object_vel = self.sim.data.get_joint_qvel('object:joint')
            object_pose = self._get_body_pose('object')

        observation = np.concatenate([
            forearm_pos, forearm_rot, forearm_velp, robot_qpos, robot_qvel, object_pose, object_vel
        ])
        return {
            'observation': observation,
            'achieved_goal': self._get_achieved_goal().ravel(),
            'desired_goal': self.goal.ravel().copy(),
        }


class HandPickAndPlaceEnv(MovingHandEnv):
    def __init__(self, reward_type='sparse', **kwargs):
        super(HandPickAndPlaceEnv, self).__init__(
            model_path=HAND_PICK_AND_PLACE_XML,
            reward_type=reward_type,
            has_object=True, **kwargs
        )


class MovingHandReachEnv(MovingHandEnv):
    def __init__(self, reward_type='sparse', **kwargs):
        super(MovingHandReachEnv, self).__init__(
            model_path=HAND_MOVE_AND_REACH_XML,
            reward_type=reward_type,
            has_object=False, **kwargs
        )
