import copy
from enum import Enum

import numpy as np
import gym
from gym import spaces
from gym.envs.robotics import hand_env
from gym.envs.robotics.hand.reach import FINGERTIP_SITE_NAMES
from gym.envs.robotics.hand.move import HandPickAndPlaceEnv, FINGERTIP_BODY_NAMES
from gym.envs.robotics.utils import reset_mocap2body_xpos
from gym.utils import transformations as tf
from gym.utils import kinematics as kin


class HandSteppedTask(Enum):
    PICK_AND_PLACE = 1
    LIFT_ABOVE_TABLE = 2


class HandSteppedEnv(gym.GoalEnv):

    def __init__(self, *, task: HandSteppedTask=None, render_substeps=False):
        super(HandSteppedEnv, self).__init__()

        self.metadata = {
            'render.modes': ['human'],
        }

        self._qp_solver = None
        self.task = task or HandSteppedTask.LIFT_ABOVE_TABLE
        self.render_substeps = render_substeps
        self.sim_env = HandPickAndPlaceEnv(weld_fingers=False, object_id='box')
        self.goal = self._sample_goal()
        obs = self._get_obs()

        if self.task == HandSteppedTask.PICK_AND_PLACE:
            # n_actions = 3 * 6
            raise NotImplementedError
        elif self.task == HandSteppedTask.LIFT_ABOVE_TABLE:
            n_actions = 3 * 5
        else:
            raise NotImplementedError

        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    # Shortcuts
    # ----------------------------

    @property
    def viewer(self):
        return self.sim_env.viewer

    @property
    def sim(self):
        return self.sim_env.sim

    def get_fingertips_pos(self):
        return np.array([self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES])

    # Env methods
    # ----------------------------

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        fingers_pos_wrt_obj = action[:(3*5)].reshape(-1, 3) * np.r_[0.03, 0.03, 0.03]

        if self.task == HandSteppedTask.LIFT_ABOVE_TABLE:
            arm_pos_wrt_world = np.zeros(3)
        else:
            arm_pos_wrt_world = action[(3*5):]

        arm_bounds = np.array(self.sim_env.forearm_bounds).T
        if self.render_substeps:
            tf.render_box(self.viewer, bounds=arm_bounds)

        arm_pos_wrt_world *= np.abs(arm_bounds[:, 1] - arm_bounds[:, 0]) / 2.0
        arm_pos_wrt_world += arm_bounds.mean(axis=1)

        obj_pose = self.sim_env._get_object_pose()
        obj_on_ground = obj_pose[2] < 0.37

        fingers_pos_targets = np.array([
            tf.apply_tf(np.r_[transf, 1., 0., 0., 0.], obj_pose) for transf in fingers_pos_wrt_obj
        ])

        self.sim.model.eq_active[1:] = 0
        # self.sim.data.mocap_pos[1:] = fingers_pos_targets[:, :3]

        if self.render_substeps:
            tf.render_pose(arm_pos_wrt_world, self.sim_env.viewer, label='arm_t')
            # for i, f in enumerate(fingers_pos_targets):
            #     tf.render_pose(f, self.sim_env.viewer, label=f'f_{i}')

        pregrasp_palm_target = fingers_pos_targets[:, :3].mean(axis=0)
        pregrasp_palm_target = np.r_[pregrasp_palm_target, 1., 0., 0., 0.]
        pregrasp_palm_target = tf.apply_tf(np.r_[-0.01, 0., 0.015, 1., 0., 0., 0.], pregrasp_palm_target)[:3]

        # move hand
        if len(self.sim_env.get_object_contact_points(other_body='robot')) == 0:
            hand_action = np.r_[0., -.5, -np.ones(18)]
            hand_action[15:] = (-1., -0.5, 1., -1., 0)
            self._move_arm(pregrasp_palm_target, hand_action=hand_action)

        # move fingers
        self._move_fingers(fingers_pos_targets, max_steps=30)

        # move arm
        stable_steps = self._move_arm(arm_pos_wrt_world, count_stable_steps=True)

        done = obj_on_ground
        obs = self._get_obs()

        if self.task == HandSteppedTask.LIFT_ABOVE_TABLE:
            info = dict()
            reward = stable_steps
        else:
            info = {'is_success': self.sim_env._is_success(obs['achieved_goal'], self.goal)}
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
        self.sim.model.eq_active[1:] = 0
        self.sim_env.reset()
        return self._get_obs()

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, *args, **kwargs):
        if self.task == HandSteppedTask.LIFT_ABOVE_TABLE:
            raise NotImplementedError
        return self.sim_env.compute_reward(*args, **kwargs)

    def _sample_goal(self):
        return self.sim_env._sample_goal()

    def _get_obs(self):
        sim_obs = self.sim_env._get_obs()
        obj_pos = self.sim_env._get_object_pose()[:3]
        fingertips_pos = self.get_fingertips_pos()
        obs = np.r_[obj_pos, fingertips_pos.ravel()]
        return {
            **sim_obs,
            'observation': obs,
        }

    # Hand and arm control
    # ----------------------------

    def _move_arm(self, grasp_center_target: np.ndarray, hand_action: np.ndarray=None,
                  threshold=0.01, k=0.1, max_steps=100, count_stable_steps=False):
        if hand_action is not None:
            hand_action = hand_action.copy()
        stable_steps = 0
        prev_rel_pos = np.zeros(3)
        reset_mocap2body_xpos(self.sim)

        for i in range(max_steps):
            grasp_center_pos = self.sim_env._get_grasp_center_pose(no_rot=True)[:3]
            d = grasp_center_target - grasp_center_pos
            if np.linalg.norm(d) < threshold and not count_stable_steps:
                break
            # set hand action
            if hand_action is not None:
                hand_env.HandEnv._set_action(self.sim_env, hand_action)
            self.sim.data.mocap_pos[0] += d * k
            self.sim.step()

            if count_stable_steps:
                obj_pos = self.sim_env._get_object_pose()[:3]
                rel_pos = obj_pos - grasp_center_pos
                still = prev_rel_pos is not None and np.all(np.abs(rel_pos - prev_rel_pos) < 0.002)
                obj_above_table = len(self.sim_env.get_object_contact_points(other_body='table')) == 0
                if still and obj_above_table:
                    stable_steps += 1
                elif i > 10:
                    break
                prev_rel_pos = rel_pos

            if self.render_substeps:
                self.render(keep_markers=True)

        if count_stable_steps:
            return stable_steps

    def _move_fingers(self, targets: np.ndarray, threshold=0.01, k=0.1, max_steps=100, multiobjective_solver=False):
        for _ in range(max_steps):
            fingers_pos_curr = self.get_fingertips_pos()
            err = np.linalg.norm(fingers_pos_curr[:, :3] - targets[:, :3])
            if err < threshold:
                break

            if multiobjective_solver:
                vels = []
                for t_pose, c_pos, body in zip(targets, fingers_pos_curr, FINGERTIP_BODY_NAMES):
                    cart_vel = np.zeros(6)
                    cart_vel[:3] = (t_pose[:3] - c_pos) * k
                    vels.append(cart_vel[:3])
                sol, opt, ctrl_idx = self._solve_hand_ik_multiobjective_vel(FINGERTIP_BODY_NAMES, vels, no_wrist=False)
                if opt:
                    self.sim.data.ctrl[:] += np.clip(sol, -.5, .5)
            else:
                for t_pose, c_pos, body in zip(targets, fingers_pos_curr, FINGERTIP_BODY_NAMES):
                    cart_vel = np.zeros(6)
                    cart_vel[:3] = (t_pose[:3] - c_pos) * k
                    sol, opt, ctrl_idx = self._solve_hand_ik_vel(body, cart_vel, no_wrist=True, check_joint_lims=False)
                    if opt:
                        self.sim.data.ctrl[ctrl_idx] += np.clip(sol, -.5, .5)

            self.sim.step()
            if self.render_substeps:
                self.render(keep_markers=True)

    # IK solvers
    # ----------------------------

    def _solve_hand_ik_vel(self, ee_body: str, cart_vel: np.ndarray, no_wrist=False,
                           check_joint_lims=True, no_rot=True):

        jac = kin.get_jacobian(self.sim.model, self.sim.data, self.sim.model.body_name2id(ee_body))
        ee_initials = ee_body.replace('robot0:', '').replace('distal', '').upper()

        jac_idx = []
        qpos = []
        ctrl_idx = []
        joint_limits = []
        for i in range(jac.shape[1]):
            jnt_id = self.sim.model.dof_jntid[i]
            if self.sim.model.jnt_type[jnt_id] != 3:
                # only rotational joints
                continue
            jnt_name = self.sim.model.joint_id2name(jnt_id)
            if ee_initials not in jnt_name and no_wrist:
                continue
            act_name = jnt_name.replace('robot0:', 'robot0:A_')
            try:
                act_id = self.sim.model.actuator_name2id(act_name)
            except ValueError:
                continue
            jac_idx.append(i)
            qpos_addr = self.sim.model.jnt_qposadr[jnt_id]
            qpos.append(self.sim.data.qpos[qpos_addr])
            ctrl_idx.append(act_id)
            joint_limits.append(self.sim.model.jnt_range[jnt_id])
        jac = jac[:, jac_idx]
        qpos = np.array(qpos)
        joint_limits = np.array(joint_limits)
        assert qpos.shape[0] == jac.shape[1]

        if not check_joint_lims:
            joint_limits = None

        if no_rot:
            cart_vel = cart_vel[:3]
            jac = jac[:3]

        sol, opt = kin.solve_qp_ik_vel(cart_vel, jac, qpos, joint_lims=joint_limits,
                                       duration=0.1, margin=0.1, solver=self._qp_solver)
        return sol, opt, ctrl_idx

    def _solve_hand_ik_multiobjective_vel(self, bodies: list, velocities: list, no_wrist=False):

        n_end_effectors = len(bodies)
        velocities = np.concatenate(velocities)
        assert velocities.shape == (n_end_effectors * 3,)

        jacobians = []
        all_qpos = None
        all_ctrl_idx = None
        for ee_body in bodies:
            jac = kin.get_jacobian(self.sim.model, self.sim.data, self.sim.model.body_name2id(ee_body))
            ee_initials = ee_body.replace('robot0:', '').replace('distal', '').upper()
            jac_idx = []
            qpos = []
            ctrl_idx = []
            for i in range(jac.shape[1]):
                jnt_id = self.sim.model.dof_jntid[i]
                if self.sim.model.jnt_type[jnt_id] != 3:
                    # only rotational joints
                    continue
                jnt_name = self.sim.model.joint_id2name(jnt_id)
                if ee_initials not in jnt_name and no_wrist:
                    continue
                act_name = jnt_name.replace('robot0:', 'robot0:A_')
                try:
                    act_id = self.sim.model.actuator_name2id(act_name)
                except ValueError:
                    continue
                jac_idx.append(i)
                qpos_addr = self.sim.model.jnt_qposadr[jnt_id]
                qpos.append(self.sim.data.qpos[qpos_addr])
                ctrl_idx.append(act_id)
            jac = jac[:, jac_idx]
            qpos = np.array(qpos)
            assert qpos.shape[0] == jac.shape[1]
            jacobians.append(jac[:3])
            all_qpos = qpos
            all_ctrl_idx = ctrl_idx

        jacobians = np.concatenate(jacobians)
        sol, opt = kin.solve_qp_ik_vel(velocities, jacobians, all_qpos, solver=self._qp_solver)
        return sol, opt, all_ctrl_idx
