import itertools as it
from collections import deque

from mujoco_py import const as mj_const
import numpy as np

import gym
from gym.utils.mjviewer import add_selection_logger


ENV = 'YumiReachTwoArms-v0'
# ENV = 'YumiReachRightArm-v0'
# ENV = 'YumiReachLeftArm-v0'


def test_sampling_time():
    import timeit
    raw_env = gym.make(ENV).unwrapped
    iters = 300
    res = timeit.timeit('raw_env._sample_goal()', number=iters, globals=locals())
    print('Samples per second:', iters/res)


def main():

    env = gym.make(ENV)
    raw_env = env.unwrapped
    sim = raw_env.sim
    env.render()
    add_selection_logger(raw_env.viewer, sim)

    env.reset()
    markers = deque(maxlen=300)

    for _ in it.count():

        obs = env.reset()
        goal = obs['desired_goal']

        if len(markers) < markers.maxlen:
            if raw_env.has_left_arm:
                markers.append(dict(
                    pos=goal[:3], label="", type=mj_const.GEOM_SPHERE,
                    size=np.ones(3)*0.05, rgba=np.r_[0.2, 0., 1., 1], specular=0.
                ))
            if raw_env.has_right_arm:
                markers.append(dict(
                    pos=goal[3:], label="", type=mj_const.GEOM_SPHERE,
                    size=np.ones(3)*0.05, rgba=np.r_[0.2, .7, 1., 1], specular=0.
                ))

        for m in markers:
            raw_env.viewer.add_marker(**m)
        env.render()


if __name__ == '__main__':
    main()
