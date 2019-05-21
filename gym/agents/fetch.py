import numpy as np

import gym
from gym.agents.base import BaseAgent
from gym.utils import transformations as tf

class FetchPickAndPlaceAgent(BaseAgent):

    def __init__(self, env, **kwargs):
        super(FetchPickAndPlaceAgent, self).__init__(env, **kwargs)
        self._phase = -1
        self._goal = None

    def predict(self, obs, **kwargs):

        # Modified from original implementation in OpenAI's baselines:
        # https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py

        goal = obs['desired_goal']
        if self._goal is None or np.any(goal != self._goal):
            self._goal = goal.copy()
            self._phase = -1

        grasp_center_pos = obs['observation'][:3]
        object_pos = obs['observation'][3:6]
        object_rel_pos = obs['observation'][6:9]

        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object

        raw_env = self._env.unwrapped
        table_tf = raw_env.get_table_surface_pose()

        print('HERE', np.linalg.norm(object_pos[:2] - table_tf[:2]))

        if raw_env.has_rotating_platform:

            if np.linalg.norm(object_pos[:2] - table_tf[:2]) > 0.22: # FIXME
                far_end_pose = np.r_[
                    raw_env.sim.data.get_site_xpos('rotating_platform:far_end'),
                    tf.rotations.mat2quat(raw_env.sim.data.get_site_xmat('rotating_platform:far_end')),
                ]

                close_end_pose = tf.apply_tf(np.r_[-0.5, -0.09, 0., 1., 0., 0., 0.], far_end_pose)
                tf.render_pose(close_end_pose, raw_env.viewer, label="close_end")

                # push_target = np.r_[0.2, 0.2, 0., 1., 0., 0., 0.]
                # push_target = tf.apply_tf(table_tf, push_target)
                push_target = tf.apply_tf(np.r_[0., 0.1, 0., 1., 0., 0., 0.], close_end_pose)
                tf.render_pose(push_target, raw_env.viewer)

                if self._phase == -1:
                    close_end_pose[2] += 0.05

                err = close_end_pose[:3] - grasp_center_pos
                u = np.zeros(4)

                if self._phase < 1:
                    if np.linalg.norm(err) > 0.015:
                        u[3] = -1.
                        u[:3] = err * 20.0
                    else:
                        self._phase += 1
                        self._phase_steps = 0

                if self._phase == 1:
                    u[3] = -1.
                    u[:3] = (push_target[:3] - grasp_center_pos) * 10
                    self._phase_steps += 1
                    if self._phase_steps > 80:
                        self._phase += 1
                        self._phase_steps = 0

                return u

            elif self._phase < 2:
                self._phase = 2
                self._phase_steps = 0

        elif self._phase < 2:
            self._phase = 2
            self._phase_steps = 0

        if np.abs(table_tf[2] - grasp_center_pos[2]) < 0.07 and self._phase == 2:
            action = [0, 0, 0.4, 0.05]
            return action
        elif self._phase == 2:
            self._phase = 3
            self._phase_steps = 0

        if np.linalg.norm(object_oriented_goal) >= 0.005 and self._phase == 3:
            action = [0, 0, 0, 0]
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[2] += 0.03
            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]*6
            action[len(action)-1] = 0.05
            return action
        elif self._phase == 3:
            self._phase = 4
            self._phase_steps = 0

        if np.linalg.norm(object_rel_pos) >= 0.005 and self._phase == 4:
            action = [0, 0, 0, 0]
            for i in range(len(object_rel_pos)):
                action[i] = object_rel_pos[i]*6
            action[len(action)-1] = -0.2
            return action
        elif self._phase == 4:
            self._phase = 5
            self._phase_steps = 0

        if np.linalg.norm(goal - object_pos) >= 0.01 and self._phase == 5:
            action = [0, 0, 0, 0]
            for i in range(len(goal - object_pos)):
                action[i] = (goal - object_pos)[i]*6
            action[len(action)-1] = -0.2
            return action
        elif self._phase == 5:
            self._phase = 6
            self._phase_steps = 0

        if self._phase == 6:
            action = [0, 0, 0, 0]
            action[len(action)-1] = -0.2 # keep the gripper closed
            return action


class FetchPushAgent(BaseAgent):

    def __init__(self, env, **kwargs):
        super(FetchPushAgent, self).__init__(env, **kwargs)
        self._goal = None
        self._phase = 0
        self._last_disp = None

    def predict(self, obs, **kwargs):

        goal = obs['desired_goal']
        if self._goal is None or np.any(goal != self._goal):
            self._goal = goal.copy()
            self._phase = 0
            self._last_disp = None

        grp_pos = obs['observation'][:3]
        object_pos = obs['observation'][3:6]
        # object_rel_pos = obs['observation'][6:9]

        disp = self._goal[:2] - object_pos[:2]
        if self._last_disp is None or np.linalg.norm(disp) > 0.05:
            self._last_disp = disp.copy()

        disp_dir = self._last_disp / np.linalg.norm(self._last_disp)

        push_goal_pos = object_pos - np.r_[disp_dir, 0.] * 0.06

        if self._phase == 0:
            push_goal_pos[2] = 0.5
        elif self._phase == 2:
            push_goal_pos[:2] += np.clip(disp*1.7, -0.08, 0.08)

        grp_disp = push_goal_pos - grp_pos

        action = np.zeros(4)
        action[-1] = -1.0

        if np.linalg.norm(disp) < 0.01:
            return action

        action[:2] = grp_disp[:2] * 8.
        action[2] = grp_disp[2] * 8.

        if np.linalg.norm(grp_disp) < 0.01:
            if self._phase == 0:
                self._phase = 1
            elif self._phase == 1:
                self._phase = 2

        return action


def _main():
    env = gym.make('FetchPickAndPlace-v1')
    # env.unwrapped.reward_params = dict(stepped=True)
    env.unwrapped.target_in_the_air = False
    agent = FetchPushAgent(env)
    obs = None
    done = True
    while True:
        if done:
            obs = env.reset()
        obs, rew, done, _ = env.step(agent.predict(obs))
        print(rew)
        env.render()


if __name__ == '__main__':
    _main()
