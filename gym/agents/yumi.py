import numpy as np


class YumiReachAgent(object):

    def __init__(self, env):
        super(YumiReachAgent).__init__()
        import PyKDL
        from gym.utils.kdl_parser import kdl_tree_from_urdf_model
        from gym.envs.yumi.yumi_env import YumiEnv
        assert isinstance(env.unwrapped, YumiEnv)
        assert env.unwrapped.block_gripper

        self._env = env
        self._raw_env = env.unwrapped # type: YumiEnv
        self._sim = self._raw_env.sim
        self._dt = env.unwrapped.dt
        self._goal = None
        self._robot = YumiEnv.get_urdf_model()
        self._kdl = PyKDL
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

    def _controller(self, err, prev_err):
        d_err = (err - prev_err) / self._dt
        prev_err[:] = err
        return -(1.0 * err + 0.05 * d_err) * 1.0

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

    def predict(self, obs):

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
                u_masked[:] = self._controller(err_q, prev_err)

        u = np.clip(u, self._env.action_space.low, self._env.action_space.high)
        return u


class YumiDummyAgent(object):

    def __init__(self, env):
        super(YumiDummyAgent).__init__()
        self._env = env
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

    def predict(self, obs):
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
