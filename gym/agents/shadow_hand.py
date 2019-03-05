import numpy as np


class HandPickAndPlaceAgent(object):

    def __init__(self, env):
        super(HandPickAndPlaceAgent).__init__()
        from gym.envs.robotics import HandPickAndPlaceEnv
        assert isinstance(env.unwrapped, HandPickAndPlaceEnv)
        self._env = env
        self._goal = None
        self._hand_ctrl = np.zeros(18)
        self._grasp_steps = 0

    def _reset(self, obs=None):
        if obs is not None:
            self._goal = obs['desired_goal'].copy()
        else:
            self._goal = None
        self._hand_ctrl = np.zeros(18)
        self._grasp_steps = 0

    def predict(self, obs):

        if self._goal is None or np.any(self._goal != obs['desired_goal']):
            self._reset(obs)

        action = np.zeros(self._env.action_space.shape)
        obj_pos = obs['achieved_goal'][:3]
        d = obj_pos - self._env.unwrapped._get_grasp_center_pose(no_rot=True)[:3]
        reached = np.linalg.norm(d) < 0.05
        on_palm = False
        dropped = obj_pos[2] < 0.40

        if dropped:
            return action * 0.0

        if reached:
            contacts = self._env.unwrapped.get_object_contact_points()
            palm_contacts = len([x for x in contacts if 'palm' in x['body1'] or 'palm' in x['body2']])
            on_palm = palm_contacts > 0

        wrist_ctrl = -1.0
        self._hand_ctrl[:] = -1.0
        self._hand_ctrl[13:] = (-1., 1., 1., -1., -1.)
        arm_pos_ctrl = d * 1.0
        if on_palm:
            self._hand_ctrl[:] = 1.0
            self._hand_ctrl[13:] = (1., 1., 1., -1., -1.)
            arm_pos_ctrl *= 0.0
            self._grasp_steps += 1
            if self._grasp_steps > 10:
                d = obs['desired_goal'][:3] - obs['achieved_goal'][:3]
                arm_pos_ctrl = d * 0.5
        else:
            self._grasp_steps = 0

        action[1] = wrist_ctrl
        action[-7:-4] = arm_pos_ctrl
        action[2:-7] = self._hand_ctrl
        return action
