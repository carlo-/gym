from time import sleep
import itertools as it

import numpy as np
import gym


def main():
    env = gym.make('HandPickAndPlace-v0')
    env.reset()

    sim = env.unwrapped.sim
    model = sim.model

    for i in it.count():

        for j in range(6):

            env.reset()
            for val in np.linspace(-1, 1, 60):

                env.render()
                action = env.action_space.sample()
                action[-7:] *= 0.0
                action[20+j] = val

                env.step(action)
                sleep(1/60)


if __name__ == '__main__':
    main()
