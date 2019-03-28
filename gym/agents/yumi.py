import copy

import numpy as np

from gym.envs.yumi.yumi_env import YumiEnv
from gym.envs.robotics import rotations
from gym.agents.base import BaseAgent
from mujoco_py import const as mj_const


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


def _simulate_mocap_ctrl(raw_env: YumiEnv, pose_delta, arm):
    prev_s = copy.deepcopy(raw_env.sim.get_state())
    mocap_a = np.zeros((raw_env.sim.model.nmocap, 7))
    if arm == 'l' or (arm == 'r' and not raw_env.has_two_arms):
        mocap_a[0] = pose_delta
    elif arm == 'r':
        mocap_a[1] = pose_delta
    else:
        raise NotImplementedError
    raw_env.mocap_control(mocap_a)
    target_qpos = raw_env.sim.data.qpos.copy()
    raw_env.sim.set_state(prev_s)
    arm_target_qpos = target_qpos[getattr(raw_env, f'_arm_{arm}_joint_idx')]
    return arm_target_qpos


def quat_angle_diff(quat_a, quat_b):
    quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
    angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
    return np.abs(rotations.normalize_angles(angle_diff))


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    rot_mat = rotations.quat2mat(pose[..., 3:])
    mat = np.zeros(pose.shape[:-1] + (4, 4))
    mat[..., :3, :3] = rot_mat
    mat[..., :3, 3] = pose[..., :3]
    mat[..., 3, 3] = 1.0
    return mat


def mat_to_pose(mat: np.ndarray) -> np.ndarray:
    rot_mat = mat[..., :3, :3]
    quat = rotations.mat2quat(rot_mat)
    pos = mat[..., :3, 3]
    return np.concatenate([pos, quat], axis=-1)


def _get_tf(world_to_b: np.ndarray, world_to_a: np.ndarray) -> np.ndarray:
    """Returns a_to_b pose"""
    pos_tf = world_to_b[..., :3] - world_to_a[..., :3]
    q = rotations.quat_mul(rotations.quat_identity(), rotations.quat_conjugate(world_to_a[..., 3:]))
    pos_tf = rotations.quat_rot_vec(q, pos_tf)
    quat_tf = rotations.quat_mul(world_to_b[..., 3:], rotations.quat_conjugate(world_to_a[..., 3:]))
    return np.concatenate([pos_tf, quat_tf], axis=-1)


def _apply_tf_old(a_to_b: np.ndarray, world_to_a: np.ndarray) -> np.ndarray:
    """Returns world_to_b pose"""
    a_to_b_mat = pose_to_mat(a_to_b)
    world_to_a_mat = pose_to_mat(world_to_a)
    world_to_b_mat = world_to_a_mat @ a_to_b_mat
    return mat_to_pose(world_to_b_mat)


def _apply_tf(a_to_b: np.ndarray, world_to_a: np.ndarray) -> np.ndarray:
    """Returns world_to_b pose"""
    new_pos = world_to_a[:3] + rotations.quat_rot_vec(world_to_a[3:], a_to_b[:3])
    new_quat = rotations.quat_mul(a_to_b[3:], world_to_a[3:])
    return np.concatenate([new_pos, new_quat], axis=-1)


def _render_pose(pose: np.ndarray, viewer, label=""):
    pos = pose[:3]
    if pose.size == 6:
        quat = rotations.euler2quat(pose[3:])
    elif pose.size == 7:
        quat = pose[3:]
    else:
        raise ValueError
    for i in range(3):
        rgba = np.zeros(4)
        rgba[[i, 3]] = 1.
        tf = [np.r_[0., np.pi/2, 0.], np.r_[np.pi/2, np.pi/1, 0.], np.r_[0., 0., 0.]][i]
        rot = rotations.quat_mul(quat, rotations.euler2quat(tf))
        viewer.add_marker(
            pos=pos, mat=rotations.quat2mat(rot).flatten(), label=(label if i == 0 else ""),
            type=mj_const.GEOM_ARROW, size=np.r_[0.01, 0.01, 0.2], rgba=rgba, specular=0.
        )


class TFDebugger:

    i = 0
    ob1_pose = None
    ob2_pose = None
    tf = None

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def reset(cls):
        cls.i = 0
        cls.ob1_pose = np.r_[0.025, 0.025, 0.825, 1, 0, 0, 0]
        cls.ob2_pose = np.r_[0.025, 0.25, 0.825, 1, 0, 0, 0]
        cls.tf = None

    @classmethod
    def step(cls, viewer):

        q1 = rotations.euler2quat(np.r_[0.01, 0.01, 0.05])
        q0 = cls.ob1_pose[3:].copy()
        q_common = rotations.quat_mul(q0, q1)

        cls.ob1_pose[3:] = q_common.copy()
        _render_pose(cls.ob1_pose, viewer)

        cls.ob2_pose[3:] = q_common.copy()
        _render_pose(cls.ob2_pose, viewer)

        if cls.i < 37:
            cls.tf = _get_tf(cls.ob1_pose, cls.ob2_pose)  # 2_to_1
            cls.i += 1
        else:
            ob1_pose_back = _apply_tf_old(cls.tf, cls.ob2_pose)
            ob1_pose_back[2] -= 0.0
            _render_pose(_apply_tf_old(cls.tf, cls.ob2_pose), viewer)


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
                u[7] = -1 + self._phase_steps / 15.
                u[15] = -1 + self._phase_steps / 15.
            elif self._phase == 3:
                u[7] = 1.0
                u[15] = 1.0

            _render_pose(target_pose.copy(), self._raw_env.viewer)
            curr_pose = curr_grp_poses[arm].copy()
            err_pose = curr_pose - target_pose
            err_pos = np.linalg.norm(err_pose[:3])
            pos_errors.append(err_pos)

            if self.use_mocap_ctrl:
                controller_k = 2.0
                err_rot = quat_angle_diff(curr_pose[3:], target_pose[3:])
                target_q = _simulate_mocap_ctrl(self._raw_env, -err_pose, arm)
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
        if self._phase == 0 and np.all(np.array(pos_errors) < 0.02) and err_rot < 0.05:
            self._phase = 1
            self._phase_steps = 0
        elif self._phase == 1 and np.all(np.array(pos_errors) < 0.007) and err_rot < 0.05:
            self._phase = 2
            self._phase_steps = 0
        elif self._phase == 2:
            if self._phase_steps > 30:
                self._phase = 3
                self._phase_steps = 0
                self._locked_l_to_r_tf = _get_tf(curr_grp_poses['r'], curr_grp_poses['l'])

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
                target_q = _simulate_mocap_ctrl(self._raw_env, pose_delta, arm)
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
