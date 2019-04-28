import itertools as it

import numpy as np

import gym
from gym.utils.mjviewer import add_selection_logger


def main():

    env = gym.make('YumiConstrained-v1', reward_type='sparse')
    raw_env = env.unwrapped
    sim = raw_env.sim
    env.render()
    add_selection_logger(raw_env.viewer, sim)

    done = True
    obs = None

    for _ in it.count():
        if done:
            obs = env.reset()

        center_pos = obs['observation'][:3]
        achieved_goal = obs['achieved_goal']
        err = achieved_goal - center_pos

        u = np.zeros(env.action_space.shape)
        u[1:4] = err * 10.0

        u = env.action_space.sample()

        obs, rew, done, _ = env.step(u)
        env.render()
        print(rew)


if __name__ == '__main__':
    main()
