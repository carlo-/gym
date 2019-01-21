from time import sleep
import itertools as it

import gym
from gym.envs.robotics import FetchEnv


def main():
    env = gym.make('FetchPickAndPlaceDense-v1')
    raw_env = env.unwrapped # type: FetchEnv
    raw_env.simplified_reward = True

    for _ in it.count():

        env.reset()

        for _ in range(100):

            env.render()
            action = env.action_space.sample()

            obs, rew, done, info = env.step(action)
            print(rew)
            sleep(1/60)


if __name__ == '__main__':
    main()
