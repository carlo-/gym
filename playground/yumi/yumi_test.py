from time import sleep
from datetime import datetime
import itertools as it

import numpy as np
import gym


def main():
    env = gym.make('Yumi-v0')
    env.reset()

    sim = env.unwrapped.sim
    model = sim.model

    for i in it.count():

        env.reset()
        action = env.action_space.sample() * 0.1

        tic = datetime.now()

        for j in range(200):

            env.render()
            env.step(action)

        toc = datetime.now()

        t = (toc - tic).total_seconds()
        fps = 200 / t
        print(f'fps: {fps}')


if __name__ == '__main__':
    main()
