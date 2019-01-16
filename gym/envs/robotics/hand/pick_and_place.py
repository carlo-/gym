import os
import numpy as np

from gym import utils, error
from gym.envs.robotics import rotations, hand_env
from gym.envs.robotics.utils import robot_get_obs

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, "
                                       "and also perform the setup instructions here: "
                                       "https://github.com/openai/mujoco-py/.)".format(e))


HAND_PICK_AND_PLACE_XML = os.path.join('hand', 'pick_and_place.xml')


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat


class _PickAndPlaceEnv(hand_env.HandEnv, utils.EzPickle):
    def __init__(self, model_path, reward_type, initial_qpos=None,
                 randomize_initial_position=True, randomize_initial_rotation=True,
                 distance_threshold=0.01, rotation_threshold=0.1, n_substeps=20, relative_control=False):

        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.reward_type = reward_type

        hand_env.HandEnv.__init__(self, model_path, n_substeps=n_substeps, initial_qpos=initial_qpos or dict(),
                                  relative_control=relative_control, arm_control=True)
        utils.EzPickle.__init__(self)

    def _get_achieved_goal(self):
        # TODO: Implement
        object_qpos = self.sim.data.get_joint_qpos('object:joint')
        assert object_qpos.shape == (7,)
        return object_qpos

    def _goal_distance(self, goal_a, goal_b):
        # TODO: Implement
        return np.zeros(3), np.zeros(3)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # TODO: Implement
        return 0.0

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, desired_goal):
        # TODO: Implement
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot
        return achieved_both

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.sim.data.set_joint_qpos('object:joint', initial_qpos)

        return True

    def _sample_goal(self):
        # TODO: Implement
        goal = np.zeros(7)
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (7,)
        self.sim.data.set_joint_qpos('target:joint', goal)
        self.sim.data.set_joint_qvel('target:joint', np.zeros(6))

        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.
        self.sim.forward()

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel('object:joint')
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }


class HandPickAndPlaceEnv(_PickAndPlaceEnv):
    def __init__(self, reward_type='sparse'):
        super(HandPickAndPlaceEnv, self).__init__(model_path=HAND_PICK_AND_PLACE_XML, reward_type=reward_type)
