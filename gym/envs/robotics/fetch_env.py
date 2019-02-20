from typing import Sequence

import mujoco_py
import numpy as np

from gym.envs.robotics import rotations, robot_env, utils
from scipy.special import huber


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def huber_loss(a, b):
    r = a - b
    delta = 1.0
    return np.sum(huber(delta, r), axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float or array with 2 elements): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.reward_params = None

        if isinstance(obj_range, Sequence):
            assert len(obj_range) == 2, obj_range[0] <= obj_range[1]
            self.obj_range = obj_range
        elif isinstance(obj_range, float) or isinstance(obj_range, int):
            assert obj_range >= 0.0
            self.obj_range = (0.0, obj_range)
        else:
            raise ValueError

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    def get_object_contact_points(self):
        if not self.has_object:
            raise NotImplementedError("Cannot get object contact points in an environment without objects!")

        sim = self.sim
        object_name = 'object0'
        object_pos = self.sim.data.get_site_xpos(object_name)
        object_rot = self.sim.data.get_site_xmat(object_name)
        contact_points = []

        # Partially from: https://gist.github.com/machinaut/209c44e8c55245c0d0f0094693053158
        for i in range(sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = sim.data.contact[i]
            body_name_1 = sim.model.body_id2name(sim.model.geom_bodyid[contact.geom1])
            body_name_2 = sim.model.body_id2name(sim.model.geom_bodyid[contact.geom2])

            if body_name_1.startswith('robot0:') and body_name_2 == object_name:

                c_force = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, c_force)

                # Compute contact point position wrt the object
                rel_contact_pos = object_rot.T @ (contact.pos - object_pos)

                contact_points.append(dict(
                    body1=body_name_1,
                    body2=body_name_2,
                    relative_pos=rel_contact_pos,
                    force=c_force
                ))

        return contact_points

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, info: dict):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)

        if self.reward_params is not None and self.has_object:

            # object and gripper positions
            obj_pos = achieved_goal  # type: np.ndarray
            target_pos = goal  # type: np.ndarray
            grp_pos = info['gripper_pos']  # type: np.ndarray
            grp_state = info['gripper_state']  # type: np.ndarray

            if self.reward_params.get('huber_loss', False):
                # shaped reward with Huber loss
                # similar to https://arxiv.org/pdf/1610.00633.pdf
                c1 = self.reward_params.get('c1', 1.0)
                c2 = self.reward_params.get('c2', 1.0)
                d1 = huber_loss(obj_pos, grp_pos)
                d2 = huber_loss(obj_pos, target_pos)
                d = c1 * d1 + c2 * d2

            else:

                min_dist = self.reward_params.get('min_dist', 0.03)
                c = self.reward_params.get('c', 0.0)
                k = self.reward_params.get('k', 1.0)
                grasp_bonus = self.reward_params.get('grasp_bonus', 0.0)

                # desired gripper position and distance to this goal
                grp_goal_d = goal_distance(grp_pos, obj_pos)

                # actual dist between fingers
                fingers_goal_d = grp_state.sum()

                if grp_goal_d < min_dist:
                    # gripper surrounding the object, clamp gripper distance bonus to avoid discontinuities
                    grp_goal_d = min_dist
                else:
                    # still away from object, keep gripper open
                    fingers_goal_d = 0.101 # ~ max finger distance

                d += fingers_goal_d * grasp_bonus
                d += (grp_goal_d ** k) * c

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        obs_dict = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

        if self.reward_params is not None:
            obs_dict['info'] = {
                'gripper_pos': grip_pos.copy(),
                'gripper_state': gripper_state.copy()
            }
        return obs_dict

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            range_min, range_max = self.obj_range
            assert range_min <= range_max

            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < range_min:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-range_max, range_max, size=2)

            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
