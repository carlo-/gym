import copy

import numpy as np

from gym.envs.yumi.yumi_env import YumiEnv
from gym.envs.robotics import rotations
from gym.agents.base import BaseAgent
import gym.utils.transformations as tf


def _solve_qp_ik_vel(vel, jac, joint_pos, joint_lims=None, duration=None, margin=0.2):
    """
    Solves the IK for a given pusher velocity using a QP solver, imposing joint limits.
    If the solution is optimal, it is guaranteed that the resulting joint velocities will not
    cause the joints to reach their limits (minus the margin) in the specified duration of time
    :param vel: desired EE velocity (6 values)
    :param jac: jacobian
    :param joint_pos: current joint positions
    :param joint_lims: matrix of joint limits; if None, limits are not imposed
    :param duration: how long the specified velocity will be kept (in seconds); if None, 2.0 is used
    :param margin: maximum absolute distance to be kept from the joint limits
    :return: tuple with the solution (as a numpy array) and with a boolean indincating if the result is optimal or not
    :type vel: np.ndarray
    :type jac: np.ndarray
    :type joint_pos: np.ndarray
    :type joint_lims: np.ndarray
    :type duration: float
    :type margin: float
    :rtype: (np.ndarray, bool)
    """

    import cvxopt
    x_len = len(joint_pos)

    P = cvxopt.matrix(np.identity(x_len))
    A = cvxopt.matrix(jac)
    b = cvxopt.matrix(vel)
    q = cvxopt.matrix(np.zeros(x_len))

    if duration is None:
        duration = 2.

    if joint_lims is None:
        G, h = None, None
    else:
        G = duration * np.identity(x_len)
        h = np.zeros(x_len)
        for i in range(x_len):
            dist_up = abs(joint_lims[i, 1] - joint_pos[i])
            dist_lo = abs(joint_lims[i, 0] - joint_pos[i])
            if dist_up > dist_lo:
                # we are closer to the lower limit
                # => must bound negative angular velocity, i.e. G_ii < 0
                h[i] = dist_lo
                G[i, i] *= -1
            else:
                # we are closer to the upper limit
                # => must bound positive angular velocity, i.e. G_ii > 0
                h[i] = dist_up
        h = cvxopt.matrix(h - margin)
        G = cvxopt.matrix(G)

    # sol = cvxopt.solvers.qp(P, q, A=A, b=b, G=G, h=h, options={'show_progress': False, 'kktreg': 1e-9}, kktsolver='ldl')
    sol = cvxopt.solvers.qp(P, q, A=A, b=b, G=G, h=h, options={'show_progress': False, 'refinement': 5})

    x = np.array(sol['x']).reshape(-1)
    optimal = sol['status'] == 'optimal'

    return x, optimal


def _solve_qp_ik_pos(current_pose, target_pose, jac, joint_pos, joint_lims=None, duration=None, margin=0.2):
    pos_delta = target_pose[:3] - current_pose[:3]
    q_diff = rotations.quat_mul(target_pose[3:], rotations.quat_conjugate(current_pose[3:]))
    ang_delta = rotations.quat2euler(q_diff)
    vel = np.r_[pos_delta, ang_delta * 2.0]
    jac += np.random.uniform(size=jac.shape) * 1e-5
    qvel, optimal = _solve_qp_ik_vel(vel, jac, joint_pos, joint_lims, duration, margin)
    qvel = np.clip(qvel, -1.0, 1.0)
    if not optimal:
        qvel *= 0.1
    new_q = joint_pos + qvel
    return new_q


class YumiConstrainedAgent(BaseAgent):

    def __init__(self, env, **kwargs):
        super(YumiConstrainedAgent, self).__init__(env, **kwargs)
        from gym.envs.yumi.yumi_constrained import YumiConstrainedEnv
        assert isinstance(env.unwrapped, YumiConstrainedEnv)

        self._raw_env = env.unwrapped # type: YumiConstrainedEnv
        self._goal = None
        self._phase = 0
        self._phase_steps = 0

    def reset(self, new_goal=None):
        self._goal = None
        self._phase = 0
        self._phase_steps = 0
        if new_goal is not None:
            self._goal = new_goal.copy()

    def predict(self, obs, **kwargs):

        u = np.zeros(self._env.action_space.shape)
        new_goal = obs['desired_goal']
        if self._goal is None or np.any(self._goal != new_goal):
            self.reset(new_goal)

        object_pos = obs['observation'][18:21]
        object_rel_pos = obs['observation'][24:27]
        c_points = self._raw_env.sim_env.get_object_contact_points()

        if self._phase == 0:
            if np.linalg.norm(object_rel_pos) > 0.01:
                u[0] = 0.0
                u[1:4] = object_rel_pos * 5.0
            else:
                self._phase += 1
                self._phase_steps = 0

        if self._phase == 1:
            if len(c_points) < 3:
                u[0] = -self._phase_steps / 10.0
                self._phase_steps += 1
            else:
                self._phase += 1
                self._phase_steps = 0

        if self._phase == 2:
            if len(c_points) > 2:
                u[0] = -1.0
                u[1:4] = (new_goal - object_pos) * 2.0
            else:
                self._phase = 0
                self._phase_steps = 0
        return u


class YumiLiftAgent(BaseAgent):

    def __init__(self, env, **kwargs):
        super(YumiLiftAgent, self).__init__(env, **kwargs)
        from gym.envs.yumi.yumi_env import YumiLiftEnv
        assert isinstance(env.unwrapped, YumiLiftEnv)

        self._raw_env = env.unwrapped # type: YumiLiftEnv
        self._sim = self._raw_env.sim
        self._dt = env.unwrapped.dt
        self._target_qs = dict()
        self._prev_err_l = np.zeros(7)
        self._prev_err_r = np.zeros(7)
        self._goal = None
        self._phase = 0
        self._phase_steps = 0
        self._object_geom_name = 'object0_base'
        self._object_size = None

    def reset(self, new_goal=None):
        self._target_qs = dict()
        self._prev_err_l = np.zeros(7)
        self._prev_err_r = np.zeros(7)
        self._goal = None
        self._phase = 0
        self._phase_steps = 0
        self._object_size = self._get_object_size()
        if new_goal is not None:
            self._goal = new_goal.copy()

    def _controller(self, err, prev_err, k=0.1):
        d_err = (err - prev_err) / self._dt
        prev_err[:] = err
        return -(1.0 * err + 0.05 * d_err) * k

    def _get_object_size(self):
        geom = self._raw_env.sim.model.geom_name2id(self._object_geom_name)
        return self._raw_env.sim.model.geom_size[geom].copy()

    def predict(self, obs, **kwargs):

        u = np.zeros(self._env.action_space.shape)
        new_goal = obs['desired_goal']
        if self._goal is None or np.any(self._goal != new_goal):
            self.reset(new_goal)

        obj_radius = self._object_size[0]
        obj_achieved_alt = obs['achieved_goal'].item()
        obj_achieved_pose = obs['observation'][44:51]
        if np.all(obj_achieved_pose[3:] == 0):
            obj_achieved_pose[3] = 1.0
        # tf.render_pose(obj_achieved_pose, self._raw_env.viewer, label="O")

        grp_xrot = 0.9 + obj_achieved_alt * 2.0

        curr_grp_poses = {a: np.r_[
            self._raw_env.sim.data.get_site_xpos(f'gripper_{a}_center'),
            rotations.mat2quat(self._raw_env.sim.data.get_site_xmat(f'gripper_{a}_center')),
        ] for a in ('l', 'r')}

        pos_errors = []
        for arm in ('l', 'r'):

            if arm == 'l':
                transf = np.r_[0., obj_radius, 0., 1., 0., 0., 0.]
                if self._phase == 3:
                    transf[1] *= 0.9
                    transf[2] = 0.05
                pose = tf.apply_tf(transf, obj_achieved_pose)
                # tf.render_pose(pose, self._raw_env.viewer, label="L")

                grp_target_pos = pose[:3]
                grp_target_rot = np.r_[np.pi - grp_xrot, 0.01, 0.01]
                target_pose = np.r_[grp_target_pos, rotations.euler2quat(grp_target_rot)]

                prev_err = self._prev_err_l
                curr_q = obs['observation'][:7]
                u_masked = u[:7]

            elif arm == 'r':
                transf = np.r_[0., -obj_radius, 0., 1., 0., 0., 0.]
                if self._phase == 3:
                    transf[1] *= 0.9
                    transf[2] = 0.05
                pose = tf.apply_tf(transf, obj_achieved_pose)
                # tf.render_pose(pose, self._raw_env.viewer, label="R")

                grp_target_pos = pose[:3]
                grp_target_rot = np.r_[-np.pi + grp_xrot, 0.01, np.pi]
                target_pose = np.r_[grp_target_pos, rotations.euler2quat(grp_target_rot)]

                prev_err = self._prev_err_r
                curr_q = obs['observation'][16:23]
                u_masked = u[8:15]
            else:
                continue

            u[7] = -1.0
            u[15] = -1.0

            if self._phase == 0:
                target_pose[2] += 0.1
                target_pose[1] += 0.05 * np.sign(target_pose[1])
            elif self._phase == 1:
                target_pose[2] += 0.01
                target_pose[1] += 0.05 * np.sign(target_pose[1])
            elif self._phase == 2:
                target_pose[2] += 0.01
            elif self._phase == 3:
                target_pose[2] += 0.00

            if self._raw_env.viewer is not None:
                tf.render_pose(target_pose.copy(), self._raw_env.viewer)

            curr_pose = curr_grp_poses[arm].copy()
            err_pose = curr_pose - target_pose
            err_pos = np.linalg.norm(err_pose[:3])
            pos_errors.append(err_pos)

            controller_k = 2.0
            err_rot = tf.quat_angle_diff(curr_pose[3:], target_pose[3:])
            target_q = self._raw_env.mocap_ik(-err_pose, arm)

            err_q = curr_q - target_q
            u_masked[:] = self._controller(err_q, prev_err, controller_k)

        self._phase_steps += 1
        if self._phase == 0 and np.all(np.array(pos_errors) < 0.03) and err_rot < 0.1:
            self._phase = 1
            self._phase_steps = 0
        elif self._phase == 1 and np.all(np.array(pos_errors) < 0.03) and err_rot < 0.1:
            self._phase = 2
            self._phase_steps = 0
        elif self._phase == 2:
            if self._phase_steps > 30:
                self._phase = 3
                self._phase_steps = 0

        u = np.clip(u, self._env.action_space.low, self._env.action_space.high)
        return u


class YumiBarAgent(BaseAgent):

    def __init__(self, env, use_qp_solver=False, check_joint_limits=False, use_mocap_ctrl=True, **kwargs):
        super(YumiBarAgent, self).__init__(env, **kwargs)
        import PyKDL
        from gym.utils.kdl_parser import kdl_tree_from_urdf_model
        from gym.envs.yumi.yumi_env import YumiEnv
        assert isinstance(env.unwrapped, YumiEnv)
        assert env.unwrapped.has_two_arms
        assert not env.unwrapped.block_gripper

        self._raw_env = env.unwrapped # type: YumiEnv
        self._sim = self._raw_env.sim
        self._dt = env.unwrapped.dt
        self._goal = None
        self._phase = 0
        self._phase_steps = 0
        self._locked_l_to_r_tf = None
        self._robot = YumiEnv.get_urdf_model()
        self._kdl = PyKDL
        self.use_mocap_ctrl = use_mocap_ctrl
        self.use_qp_solver = use_qp_solver
        self.check_joint_limits = check_joint_limits

        if check_joint_limits and not use_qp_solver:
            raise ValueError('Joint limits can only be checked with the QP solver!')

        # make sure the simulator is ready before getting the transforms
        self._sim.forward()
        self._base_offset = self._sim.data.get_body_xpos('yumi_base_link').copy()

        kdl_tree = kdl_tree_from_urdf_model(self._robot)
        base_link = self._robot.get_root()
        ee_arm_chain_r = kdl_tree.getChain(base_link, 'gripper_r_base')
        ee_arm_chain_l = kdl_tree.getChain(base_link, 'gripper_l_base')
        if self._raw_env.has_right_arm:
            self._add_site_to_chain(ee_arm_chain_r, 'gripper_r_center')
        if self._raw_env.has_left_arm:
            self._add_site_to_chain(ee_arm_chain_l, 'gripper_l_center')

        self._fkp_solver_r = PyKDL.ChainFkSolverPos_recursive(ee_arm_chain_r)
        self._fkp_solver_l = PyKDL.ChainFkSolverPos_recursive(ee_arm_chain_l)

        self._ikp_solver_r = PyKDL.ChainIkSolverPos_LMA(ee_arm_chain_r)
        self._ikp_solver_l = PyKDL.ChainIkSolverPos_LMA(ee_arm_chain_l)

        self._jac_solver_r = PyKDL.ChainJntToJacSolver(ee_arm_chain_r)
        self._jac_solver_l = PyKDL.ChainJntToJacSolver(ee_arm_chain_l)

        # self._ikv_solver_r = PyKDL.ChainIkSolverVel_wdls(ee_arm_chain_r)
        # self._ikv_solver_l = PyKDL.ChainIkSolverVel_wdls(ee_arm_chain_l)
        # self._ikp_solver_r = PyKDL.ChainIkSolverPos_NR(ee_arm_chain_r, self._fkp_solver_r, self._ikv_solver_r)
        # self._ikp_solver_l = PyKDL.ChainIkSolverPos_NR(ee_arm_chain_l, self._fkp_solver_l, self._ikv_solver_l)

    def _add_site_to_chain(self, ee_arm_chain, site_name):
        sim = self._env.unwrapped.sim
        site_id = sim.model.site_name2id(site_name)
        pos = sim.model.site_pos[site_id]
        quat = sim.model.site_quat[site_id] # w x y z
        quat = np.r_[quat[1:], quat[3]] # x y z w

        joint = self._kdl.Joint(site_name, getattr(self._kdl.Joint, 'None'))
        frame = self._kdl.Frame(self._kdl.Rotation.Quaternion(*quat), self._kdl.Vector(*pos))

        segment = self._kdl.Segment(joint, frame)
        ee_arm_chain.addSegment(segment)

    def _controller(self, err, prev_err, k=0.1):
        d_err = (err - prev_err) / self._dt
        prev_err[:] = err
        return -(1.0 * err + 0.05 * d_err) * k

    def _reset(self, new_goal):
        self._target_qs = dict()
        self._prev_err_l = np.zeros(7)
        self._prev_err_r = np.zeros(7)
        self._goal = new_goal.copy()
        self._phase = 0
        self._phase_steps = 0
        self._locked_l_to_r_tf = None

    def _get_jacobian(self, joint_pos, jac_solver):
        seed_array = self._kdl.JntArray(len(joint_pos))
        for idx, val in enumerate(joint_pos):
            seed_array[idx] = val
        jacobian = self._kdl.Jacobian(len(joint_pos))
        jac_solver.JntToJac(seed_array, jacobian)
        jac_np = np.zeros([int(jacobian.rows()), int(jacobian.columns())])
        for i in range(int(jacobian.rows())):
            for j in range(int(jacobian.columns())):
                jac_np[i, j] = jacobian[i, j]
        return jac_np

    def _position_ik_qp(self, current_pose: np.ndarray, target_pose: np.ndarray, current_q: np.ndarray, jac_solver, joint_lims):
        jac = self._get_jacobian(current_q, jac_solver)
        sol = _solve_qp_ik_pos(current_pose, target_pose, jac, current_q, joint_lims, duration=self._raw_env.dt)
        return sol

    def _position_ik(self, pose: np.ndarray, current_q: np.ndarray, ikp_solver):

        pos = self._kdl.Vector(*pose[:3])
        if len(pose) == 7:
            quat = np.r_[pose[6], pose[3:6]]
            rot = self._kdl.Rotation().Quaternion(*quat)
        else:
            rot = self._kdl.Rotation().RPY(*pose[3:])
        seed_array = self._kdl.JntArray(len(current_q))

        for idx, val in enumerate(current_q):
            seed_array[idx] = val

        goal_pose = self._kdl.Frame(rot, pos)
        result_angles = self._kdl.JntArray(len(current_q))

        if ikp_solver.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            return np.fromiter(result_angles, dtype=np.float)
        else:
            # no solution found!
            return None

    def _position_fk(self, current_q: np.ndarray, fkp_solver):
        seed_array = self._kdl.JntArray(len(current_q))
        for idx, val in enumerate(current_q):
            seed_array[idx] = val
        end_frame = self._kdl.Frame()
        fkp_solver.JntToCart(seed_array, end_frame)
        pos = np.fromiter(end_frame.p, dtype=np.float)
        quat = np.array(end_frame.M.GetQuaternion())
        quat = np.r_[quat[3], quat[:3]]
        return np.r_[pos, quat]

    def predict(self, obs, **kwargs):

        u = np.zeros(self._env.action_space.shape)
        new_goal = obs['desired_goal']
        if self._goal is None or np.any(self._goal != new_goal):
            self._reset(new_goal)
            # TFDebugger.reset()
        # TFDebugger.step(self._raw_env.viewer)

        curr_grp_poses = {a: np.r_[
            self._raw_env.sim.data.get_site_xpos(f'gripper_{a}_center'),
            rotations.mat2quat(self._raw_env.sim.data.get_site_xmat(f'gripper_{a}_center')),
        ] for a in ('l', 'r')}

        pos_errors = []
        for arm in ('l', 'r'):
            if arm == 'l':
                solver = self._ikp_solver_l
                jac_solver = self._jac_solver_l
                joint_lims = self._raw_env.arm_l_joint_lims

                achieved_pos = obs['observation'][32:35]
                obj_l_rel_pos = obs['observation'][44:47]
                obj_pos = obj_l_rel_pos + achieved_pos

                target_pose = np.r_[obj_pos, np.pi - 0.9, 0.01, 0.01]
                target_pose = np.r_[target_pose[:3], rotations.euler2quat(target_pose[3:])]

                if self._phase == 3:
                    target_pose[:3] = self._raw_env.sim.data.get_site_xpos('target0:left').copy()
                    q = rotations.mat2quat(self._raw_env.sim.data.get_site_xmat('target0:left'))
                    target_pose[3:] = rotations.quat_mul(target_pose[3:], q)

                prev_err = self._prev_err_l
                curr_q = obs['observation'][:7]
                u_masked = u[:7]
            elif arm == 'r':
                solver = self._ikp_solver_r
                jac_solver = self._jac_solver_r
                joint_lims = self._raw_env.arm_r_joint_lims

                achieved_pos = obs['observation'][35:38]
                obj_r_rel_pos = obs['observation'][47:50]
                obj_pos = obj_r_rel_pos + achieved_pos

                target_pose = np.r_[obj_pos, -np.pi + 0.9, 0.01, np.pi]
                target_pose = np.r_[target_pose[:3], rotations.euler2quat(target_pose[3:])]

                if self._phase == 3:
                    target_pose[:3] = self._raw_env.sim.data.get_site_xpos('target0:right').copy()
                    q = rotations.mat2quat(self._raw_env.sim.data.get_site_xmat('target0:right'))
                    target_pose[3:] = rotations.quat_mul(q, target_pose[3:])

                prev_err = self._prev_err_r
                curr_q = obs['observation'][16:23]
                u_masked = u[8:15]
            else:
                continue

            if self._phase == 0:
                u[7] = -1.0
                u[15] = -1.0
                target_pose[2] += 0.1
            elif self._phase == 1:
                u[7] = -1.0
                u[15] = -1.0
                target_pose[2] -= 0.002
            elif self._phase == 2:
                u[7] = -1 + self._phase_steps / 3.
                u[15] = -1 + self._phase_steps / 3.
            elif self._phase == 3:
                u[7] = 1.0
                u[15] = 1.0

            if self._raw_env.viewer is not None:
                tf.render_pose(target_pose.copy(), self._raw_env.viewer)

            curr_pose = curr_grp_poses[arm].copy()
            err_pose = curr_pose - target_pose
            err_pos = np.linalg.norm(err_pose[:3])
            pos_errors.append(err_pos)

            if self.use_mocap_ctrl:
                if self._phase < 1:
                    controller_k = 3.0
                else:
                    controller_k = 2.5
                err_rot = tf.quat_angle_diff(curr_pose[3:], target_pose[3:])
                target_q = self._raw_env.mocap_ik(-err_pose, arm)
            else:
                controller_k = 0.1
                err_rot = 0.0
                target_pose[:3] -= self._base_offset
                if self.use_qp_solver:
                    if not self.check_joint_limits:
                        joint_lims = None
                    curr_pose[:3] -= self._base_offset
                    target_q = self._position_ik_qp(curr_pose, target_pose, curr_q, jac_solver, joint_lims)
                else:
                    target_q = self._position_ik(target_pose, curr_q, solver)

            if target_q is not None:
                err_q = curr_q - target_q
                u_masked[:] = self._controller(err_q, prev_err, controller_k)

        self._phase_steps += 1
        if self._phase == 0 and np.all(np.array(pos_errors) < 0.03) and err_rot < 0.05:
            self._phase = 1
            self._phase_steps = 0
        elif self._phase == 1 and np.all(np.array(pos_errors) < 0.007) and err_rot < 0.05:
            self._phase = 2
            self._phase_steps = 0
        elif self._phase == 2:
            if self._phase_steps > 6:
                self._phase = 3
                self._phase_steps = 0
                self._locked_l_to_r_tf = tf.get_tf(curr_grp_poses['r'], curr_grp_poses['l'])

        u = np.clip(u, self._env.action_space.low, self._env.action_space.high)
        return u


class YumiReachAgent(BaseAgent):

    def __init__(self, env, use_mocap_ctrl=True, **kwargs):
        super(YumiReachAgent, self).__init__(env, **kwargs)
        import PyKDL
        from gym.utils.kdl_parser import kdl_tree_from_urdf_model
        assert isinstance(env.unwrapped, YumiEnv)
        assert env.unwrapped.block_gripper

        self._env = env
        self._raw_env = env.unwrapped # type: YumiEnv
        self._sim = self._raw_env.sim
        self._dt = env.unwrapped.dt
        self._goal = None
        self._robot = YumiEnv.get_urdf_model()
        self._kdl = PyKDL
        self.use_mocap_ctrl = use_mocap_ctrl

        # make sure the simulator is ready before getting the transforms
        self._sim.forward()
        self._base_offset = self._sim.data.get_body_xpos('yumi_base_link').copy()

        kdl_tree = kdl_tree_from_urdf_model(self._robot)
        base_link = self._robot.get_root()
        ee_arm_chain_r = kdl_tree.getChain(base_link, 'gripper_r_base')
        ee_arm_chain_l = kdl_tree.getChain(base_link, 'gripper_l_base')
        if self._raw_env.has_right_arm:
            self._add_site_to_chain(ee_arm_chain_r, 'gripper_r_center')
        if self._raw_env.has_left_arm:
            self._add_site_to_chain(ee_arm_chain_l, 'gripper_l_center')

        self._ikp_solver_r = PyKDL.ChainIkSolverPos_LMA(ee_arm_chain_r)
        self._ikp_solver_l = PyKDL.ChainIkSolverPos_LMA(ee_arm_chain_l)
        self._fkp_solver_r = PyKDL.ChainFkSolverPos_recursive(ee_arm_chain_r)
        self._fkp_solver_l = PyKDL.ChainFkSolverPos_recursive(ee_arm_chain_l)

    def _add_site_to_chain(self, ee_arm_chain, site_name):
        sim = self._env.unwrapped.sim
        site_id = sim.model.site_name2id(site_name)
        pos = sim.model.site_pos[site_id]
        quat = sim.model.site_quat[site_id] # w x y z
        quat = np.r_[quat[1:], quat[3]] # x y z w

        joint = self._kdl.Joint(site_name, getattr(self._kdl.Joint, 'None'))
        frame = self._kdl.Frame(self._kdl.Rotation.Quaternion(*quat), self._kdl.Vector(*pos))

        segment = self._kdl.Segment(joint, frame)
        ee_arm_chain.addSegment(segment)

    def _controller(self, err, prev_err, k=1.0):
        d_err = (err - prev_err) / self._dt
        prev_err[:] = err
        return -(1.0 * err + 0.05 * d_err) * k

    def _reset(self, new_goal):
        self._target_qs = dict()
        self._prev_err_l = np.zeros(7)
        self._prev_err_r = np.zeros(7)
        self._goal = new_goal.copy()

    def _position_ik(self, pose: np.ndarray, current_q: np.ndarray, ikp_solver):

        pos = self._kdl.Vector(*pose[:3])
        if len(pose) == 7:
            rot = self._kdl.Rotation().Quaternion(*pose[3:])
        else:
            rot = self._kdl.Rotation().RPY(*pose[3:])
        seed_array = self._kdl.JntArray(len(current_q))

        for idx, val in enumerate(current_q):
            seed_array[idx] = val

        goal_pose = self._kdl.Frame(rot, pos)
        result_angles = self._kdl.JntArray(len(current_q))

        if ikp_solver.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            return np.fromiter(result_angles, dtype=np.float)
        else:
            # no solution found!
            return None

    def _position_fk(self, current_q: np.ndarray, fkp_solver):
        seed_array = self._kdl.JntArray(len(current_q))
        for idx, val in enumerate(current_q):
            seed_array[idx] = val
        end_frame = self._kdl.Frame()
        fkp_solver.JntToCart(seed_array, end_frame)
        pos = np.fromiter(end_frame.p, dtype=np.float)
        quat = np.array(end_frame.M.GetQuaternion())
        return np.r_[pos, quat]

    def predict(self, obs, **kwargs):

        u = np.zeros(self._env.action_space.shape)
        new_goal = obs['desired_goal']
        if self._goal is None or np.any(self._goal != new_goal):
            self._reset(new_goal)

        for arm in ('l', 'r'):
            if arm == 'l' and self._raw_env.has_left_arm:
                solver = self._ikp_solver_l
                target_pos = new_goal[:3]
                achieved_pos = obs['achieved_goal'][:3]
                prev_err = self._prev_err_l
                curr_q = obs['observation'][:7]
                u_masked = u[:7]
            elif arm == 'r' and self._raw_env.has_right_arm:
                solver = self._ikp_solver_r
                target_pos = new_goal[3:]
                achieved_pos = obs['achieved_goal'][3:]
                prev_err = self._prev_err_r
                curr_q = obs['observation'][:7]
                u_masked = u[:7]
                if self._raw_env.has_two_arms:
                    curr_q = obs['observation'][14:21]
                    u_masked = u[7:]
            else:
                continue

            if self.use_mocap_ctrl:
                controller_k = 2.0
                pose_delta = np.r_[target_pos - achieved_pos, 0., 0., 0., 0.]
                target_q = self._raw_env.mocap_ik(pose_delta, arm)
            else:
                controller_k = 1.0
                err_pos = np.linalg.norm(achieved_pos - target_pos)
                target_q = self._target_qs.get(arm)
                if target_q is None:
                    target_pose = np.zeros(6)
                    target_pose[:3] = target_pos - self._base_offset
                    target_q = self._position_ik(target_pose, curr_q, solver)
                    self._target_qs[arm] = target_q

                if err_pos > 0.04:
                    self._target_qs[arm] = None

            if target_q is not None:
                err_q = curr_q - target_q
                u_masked[:] = self._controller(err_q, prev_err, controller_k)

        u = np.clip(u, self._env.action_space.low, self._env.action_space.high)
        return u


class YumiDummyAgent(BaseAgent):

    def __init__(self, env, **kwargs):
        super(YumiDummyAgent, self).__init__(env, **kwargs)
        self._dt = env.unwrapped.dt
        self._prev_err1 = np.zeros(7)
        self._prev_err2 = np.zeros(7)
        self._goal = None

    def _controller(self, err, prev_err):
        d_err = (err - prev_err) / self._dt
        return -(1.0 * err + 0.05 * d_err) * 1.0, err.copy()

    def _reset(self, new_goal):
        self._prev_err1 = np.zeros(7)
        self._prev_err2 = np.zeros(7)
        self._goal = new_goal.copy()

    def predict(self, obs, **kwargs):
        new_goal = obs['desired_goal']
        if self._goal is None or np.any(self._goal != new_goal):
            self._reset(new_goal)

        a = self._env.action_space.sample() * 0.0
        sim = self._env.unwrapped.sim

        a1, self._prev_err1 = self._controller(sim.data.qpos[:7], self._prev_err1)
        a[:7] = a1

        if self._env.unwrapped.arm == 'both':
            a2, self._prev_err2 = self._controller(sim.data.qpos[9:16], self._prev_err2)
            if self._env.unwrapped.block_gripper:
                a[7:] = a2
            else:
                a[8:-1] = a2
        return a
