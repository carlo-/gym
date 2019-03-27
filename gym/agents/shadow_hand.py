import numpy as np

from gym.agents.base import BaseAgent


class HandPickAndPlaceAgent(BaseAgent):

    def __init__(self, env, **kwargs):
        super(HandPickAndPlaceAgent, self).__init__(env, **kwargs)
        from gym.envs.robotics import HandPickAndPlaceEnv
        assert isinstance(env.unwrapped, HandPickAndPlaceEnv)
        self._env = env
        self._goal = None
        self._prev_d = np.zeros(3)
        self._hand_ctrl = np.zeros(18)
        self._grasp_steps = 0
        self._strategy = 0
        if env.unwrapped.object_id == 'sphere':
            self._strategy = 1
        if env.unwrapped.object_id == 'small_box':
            self._strategy = 2

    def _reset(self, obs=None):
        if obs is not None:
            self._goal = obs['desired_goal'].copy()
        else:
            self._goal = None
        self._hand_ctrl = np.zeros(18)
        self._prev_d = np.zeros(3)
        self._grasp_steps = 0

    def predict(self, obs, **kwargs):

        if self._goal is None or np.any(self._goal != obs['desired_goal']):
            self._reset(obs)

        strategy = self._strategy
        action = np.zeros(self._env.action_space.shape)
        obj_pos = obs['achieved_goal'][:3]
        d = obj_pos - self._env.unwrapped._get_grasp_center_pose(no_rot=True)[:3]
        reached = np.linalg.norm(d) < 0.05
        on_palm = False
        dropped = obj_pos[2] < 0.38

        if dropped:
            return action * 0.0

        if reached:
            contacts = self._env.unwrapped.get_object_contact_points()
            palm_contacts = len([x for x in contacts if 'palm' in x['body1'] or 'palm' in x['body2']])
            on_palm = palm_contacts > 0

        if strategy == 0:
            wrist_ctrl = -1.0
            self._hand_ctrl[:] = -1.0

            self._hand_ctrl[[0, 3]] = 1.0
            self._hand_ctrl[[6, 9]] = -1.0

            self._hand_ctrl[13:] = (-1., -0.5, 1., -1., 0)

            d += np.r_[0., -0.030, 0.0]
            still = np.linalg.norm(d - self._prev_d) < 0.002
            if self._grasp_steps > 10:
                still = np.linalg.norm(d - self._prev_d) < 0.005
            self._prev_d = d.copy()

            arm_pos_ctrl = d * 1.0
            if on_palm or still:

                self._hand_ctrl[:] = 1.0
                self._hand_ctrl[[0, 3]] = 1.0
                self._hand_ctrl[[6, 9]] = -1.0
                self._hand_ctrl[13:] = (0.1, 0.5, 1., -1., 0)

                arm_pos_ctrl *= 0.0
                self._grasp_steps += 1
                if self._grasp_steps > 10:
                    d = obs['desired_goal'][:3] - obs['achieved_goal'][:3]
                    arm_pos_ctrl = d * 0.5
            else:
                self._grasp_steps = 0

        elif strategy in [1, 2]:

            d += np.r_[0., -0.035, 0.025]
            reached = np.linalg.norm(d) < 0.02
            if self._grasp_steps > 10:
                reached = np.linalg.norm(d) < 0.04

            wrist_ctrl = 0.0
            self._hand_ctrl[:] = -1.0
            self._hand_ctrl[13:] = (-1., 1., 1., -1., -1.)
            arm_pos_ctrl = d * 1.0
            if reached:
                arm_pos_ctrl *= 0.0
                self._grasp_steps += 1

                if strategy == 1:
                    self._hand_ctrl[13:] = (-0.5, 1., 1., -1., -1.)
                    self._hand_ctrl[4] = 0.6
                    self._hand_ctrl[5] = 0.5

                if strategy == 2:
                    self._hand_ctrl[13:] = (-0.1, 1., 1., -1., -1.)
                    self._hand_ctrl[6] = -0.7
                    self._hand_ctrl[7] = 0.6
                    self._hand_ctrl[8] = 0.5

                if self._grasp_steps > 10:
                    d = obs['desired_goal'][:3] - obs['achieved_goal'][:3]
                    arm_pos_ctrl = d * 0.5
            else:
                self._grasp_steps = 0
        else:
            raise NotImplementedError

        action[1] = wrist_ctrl
        action[-7:-4] = arm_pos_ctrl
        action[2:-7] = self._hand_ctrl
        return action
