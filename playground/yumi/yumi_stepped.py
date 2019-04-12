import itertools as it

import numpy as np

import gym
from gym.utils.mjviewer import add_selection_logger


def main():

    env = gym.make('YumiStepped-v1', render_substeps=True)
    raw_env = env.unwrapped
    sim = raw_env.sim
    env.render()
    add_selection_logger(raw_env.viewer, sim)

    done = True
    obs = None

    for _ in it.count():
        if done:
            obs = env.reset()

        a = 0.3

        u = np.r_[
            np.sin(a), np.cos(a), 0., 10.,
            np.sin(a + np.pi), np.cos(a + np.pi), 0., -10.,
        ]
        obs, rew, done, _ = env.step(u)
        env.render()
        print(rew)


if __name__ == '__main__':
    main()
