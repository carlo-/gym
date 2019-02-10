import itertools as it

import numpy as np
import gym


def main():
    # env = gym.make('HandPickAndPlaceDense-v0')
    env = gym.make('MovingHandReachDense-v0')
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

                rew = env.step(action)[1]
                print(rew)


if __name__ == '__main__':
    main()
