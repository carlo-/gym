import copy
from typing import Sequence

import mujoco_py
import numpy as np

from gym.utils import transformations as tf
from gym.envs.robotics import rotations, robot_env, utils
from scipy.special import huber


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def huber_loss(a, b):
    r = a - b
    delta = 1.0
    return np.sum(huber(delta, r), axis=-1)


OBJECTS = dict(
    original=dict(type='box', size='0.025 0.025 0.025', mass=2.0),
    fetch_box=dict(type='box', size='0.025 0.025 0.055', mass=2.0), # FIXME
)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, reward_params=None, explicit_goal_distance=False,
        has_rotating_platform=False, has_button=False, object_id=None, has_object_box=False,
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
            reward_params (dict): parameters for custom reward functions
            explicit_goal_distance (boolean): whether or not the observations should include the distance to the goal
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.reward_params = reward_params
        self.explicit_goal_distance = explicit_goal_distance
        self.has_rotating_platform = has_rotating_platform
        self.has_object_box = has_object_box
        self.has_button = has_button
        self.object_id = object_id
        self._button_pressed = False
        self._object_xy_pos_to_sync = None

        if isinstance(obj_range, Sequence):
            assert len(obj_range) == 2, obj_range[0] <= obj_range[1]
            self.obj_range = obj_range
        elif isinstance(obj_range, float) or isinstance(obj_range, int):
            assert obj_range >= 0.0
            self.obj_range = (0.0, obj_range)
        else:
            raise ValueError

        xml_format = None
        if 'pick_and_place.xml' in model_path:
            xml_format = dict(rotating_platform="", button="", object_box="", exclude_contacts="")
            if has_rotating_platform:
                initial_qpos['rotating_platform_joint'] = -0.75
                xml_format['rotating_platform'] = """
                <body name="rotating_platform" pos="0.1 -0.2 0.21">
                    <inertial pos="0 0 0" mass="2" diaginertia="0.1 0.1 0.1" />
                    <joint type="hinge" name="rotating_platform_joint" damping="0.8" axis="0 0 1" limited="false"/>
                    <geom pos="0 0 0" rgba="0 0.5 0 1" size="0.3 0.05 0.01" type="box" friction="1 0.95 0.01"/>
                    
                    <geom pos="0.295 0 0.01" rgba="0.5 0 0 1" size="0.0075 0.05 0.009" type="box" friction="1 0.95 0.01"/>
                    <geom pos="0.24 0.045 0.01" rgba="1 0 0 1" size="0.05 0.0075 0.009" type="box" friction="1 0.95 0.01"/>
                    <geom pos="0.24 -0.045 0.01" rgba="0 0 1 1" size="0.05 0.0075 0.009" type="box" friction="1 0.95 0.01"/>
                    
                    <site name="rotating_platform:far_end" pos="0.25 0 0"
                          size="0.02 0.02 0.02" rgba="0 0 1 0.5" type="sphere"/>
                </body>
                """

            if has_object_box:
                xml_format['object_box'] = """
                <body name="object_box" pos="0 0 0.25">
                    <geom pos="0 0.035 0" rgba="0 0.5 0 1" size="0.1 0.005 0.05" type="box" solimp="0.99 0.99 0.01" solref="0.01 1"/>
                    <geom pos="0 -0.035 0" rgba="1 0 0 1" size="0.1 0.005 0.05" type="box" solimp="0.99 0.99 0.01" solref="0.01 1"/>
                    <site name="object_box:near_end" pos="-0.08 0 -0.03"
                          size="0.02 0.02 0.02" rgba="0 0 1 0.5" type="sphere"/>
                </body>
                """
                xml_format['exclude_contacts'] = """
                <exclude body1="robot0:r_gripper_finger_link" body2="object_box"></exclude>
                <exclude body1="robot0:l_gripper_finger_link" body2="object_box"></exclude>
                <exclude body1="robot0:gripper_link" body2="object_box"></exclude>
                """

            if has_button:
                xml_format['button'] = """
                <body name="button" pos="-0.15 0 0.22">
                    <inertial pos="0 0 0" mass="2" diaginertia="0.1 0.1 0.1" />
                    <geom pos="0 0 0" rgba="0 0.5 0 1" size="0.05 0.05 0.01" type="box"/>
                    <geom name="button_geom" pos="0 0 0.01" rgba="1 0 0 1" size="0.02 0.02 0.01" type="box"/>
                </body>
                """

            if object_id is None or object_id == 'original':
                object_xml = """
                <body name="object0" pos="0.025 0.025 0.025">
                    <joint name="object0:joint" type="free" damping="0.01"></joint>
                    <geom size="0.025 0.025 0.025" type="box" condim="3" name="object0" material="block_mat" mass="2"></geom>
                    <site name="object0" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"></site>
                </body>
                """
            else:
                obj = dict(OBJECTS[object_id])
                if 'mass' not in obj.keys():
                    obj['mass'] = 0.2
                props = " ".join([f'{k}="{v}"' for k, v in obj.items()])
                object_xml = f"""
                <body name="object0" pos="0.025 0.025 0.025">
                    <joint name="object0:joint" type="free" damping="0.01"/>
                    <geom {props} condim="4" name="object0_base" material="block_mat" solimp="0.99 0.99 0.01" solref="0.01 1"/>
                    <site name="object0" pos="0 0 0" size="0.02 0.02 0.02" rgba="0 0 1 1" type="sphere"/>
                </body>
                """
            xml_format['object'] = object_xml

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos, xml_format=xml_format)

    def is_pressing_button(self):
        if not self.has_button:
            return False

        sim = self.sim
        for i in range(sim.data.ncon):

            contact = sim.data.contact[i]
            body_name_1 = sim.model.body_id2name(sim.model.geom_bodyid[contact.geom1])
            body_name_2 = sim.model.body_id2name(sim.model.geom_bodyid[contact.geom2])
            geom_name_1 = sim.model.geom_id2name(contact.geom1)
            geom_name_2 = sim.model.geom_id2name(contact.geom2)

            if 'robot0:' in body_name_1 and 'button_geom' == geom_name_2 or \
               'robot0:' in body_name_2 and 'button_geom' == geom_name_1:
                return True

        return False

    def sync_object_init_pos(self, pos: np.ndarray, wrt_table=False, now=False):
        assert pos.size == 2
        if wrt_table:
            pose = tf.apply_tf(
                np.r_[pos, 0., 1., 0., 0., 0.],
                self.get_table_surface_pose()
            )
            self._object_xy_pos_to_sync = pose[:2]
        else:
            self._object_xy_pos_to_sync = pos.copy()

        if now:
            object_qpos = self.sim.data.get_joint_qpos('object0:joint').copy()
            object_qpos[:2] = self._object_xy_pos_to_sync
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            self.sim.forward()

    def sync_goal(self, goal: np.ndarray, wrt_table=False):
        goal = goal.copy()
        if wrt_table:
            if goal.size == 3:
                goal = np.r_[goal, 1., 0., 0., 0.]
            goal = tf.apply_tf(
                goal,
                self.get_table_surface_pose()
            )
            self.goal = goal[:3]
        else:
            self.goal = goal[:3]

    def get_table_surface_pose(self):
        pose = np.r_[
            self.sim.data.get_body_xpos('table0'),
            self.sim.data.get_body_xquat('table0'),
        ]
        geom = self.sim.model.geom_name2id('table0_geom')
        size = self.sim.model.geom_size[geom].copy()
        pose[2] += size[2]
        return pose

    def get_object_contact_points(self, other_body='robot0:'):
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

            if (other_body in body_name_1 and body_name_2 == object_name) or \
               (other_body in body_name_2 and body_name_1 == object_name):

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

    def _reset_button(self):
        if self.has_button:
            self._button_pressed = False
            self.sim.model.body_pos[self.sim.model.body_name2id("button"), :] = (0.0, 0.0, 0.21)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal: np.ndarray, goal: np.ndarray, info: dict):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        success = (d < self.distance_threshold).astype(np.float32)

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

            elif self.reward_params.get('stepped', False):
                assert self.reward_type == 'dense'
                # actual dist between fingers
                fingers_dist = grp_state.sum()
                grp_close_to_obj = (goal_distance(grp_pos, obj_pos) < 0.12).astype(np.float32)
                obj_close_to_goal = (d < 0.12).astype(np.float32)
                grp_around_obj = (goal_distance(grp_pos, obj_pos) < 0.04).astype(np.float32)
                grp_above_table = float(grp_pos[2] > 0.43)
                obj_above_table = float(obj_pos[2] > 0.427)
                grasped = float(abs(0.05 - fingers_dist) < 0.005 and len(self.get_object_contact_points()) > 2)

                d = -(
                    grp_above_table * 0.05 +
                    grp_close_to_obj * 0.4 +
                    grp_around_obj * 0.7 +
                    grasped * 1.0 +
                    grasped * obj_close_to_goal * 0.5 +
                    grasped * obj_above_table * 0.4 +
                    success * 10.0
                )

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
            weights = (info or dict()).get('weights')
            if weights is not None:
                success *= weights
            return success - 1
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

        if self.is_pressing_button():
            self._did_press_button()

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
        desired_goal = self.goal.copy()

        explicit_goal_d = np.zeros(0)
        if self.explicit_goal_distance:
            explicit_goal_d = desired_goal - achieved_goal

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel, explicit_goal_d,
        ])

        obs_dict = {'observation': obs, 'achieved_goal': achieved_goal, 'desired_goal': desired_goal}
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
        self.sim.set_state(copy.deepcopy(self.initial_state))
        self.sim.forward()

        # Randomize start position of object.
        if self.has_object:
            range_min, range_max = self.obj_range
            assert range_min <= range_max

            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < range_min:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-range_max, range_max, size=2)

            self._object_xy_pos_to_sync = object_xpos.copy()
            object_qpos = self.sim.data.get_joint_qpos('object0:joint').copy()

            if self.has_rotating_platform:
                object_qpos[2] += 0.020
                object_qpos[:2] = self.sim.data.get_site_xpos('rotating_platform:far_end')[:2]
            elif self.has_button:
                object_qpos[:2] = 1.530, 0.420
            elif self.has_object_box:
                object_qpos[:2] = self.sim.data.get_site_xpos('object_box:near_end')[:2]
            else:
                object_qpos[:2] = object_xpos

            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self._reset_button()
        self.sim.forward()
        return True

    def _did_press_button(self):
        if self._button_pressed:
            return
        self._button_pressed = True

        # reset object position
        self.sync_object_init_pos(self._object_xy_pos_to_sync, now=True)

        # hide button away
        self.sim.model.body_pos[self.sim.model.body_name2id("button"), :] = (2., 2., 2.)
        self.sim.forward()

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
