from time import sleep
import itertools as it

import numpy as np
import gym


def ctrl_for_arm_pose(model, pose: np.ndarray):
    assert len(pose) == 6
    ctrl_range = model.actuator_ctrlrange[20:]
    actuation_range = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.
    actuation_center = (ctrl_range[:, 1] + ctrl_range[:, 0]) / 2.
    return (pose - actuation_center) / actuation_range


def main():
    env = gym.make('HandPickAndPlace-v0')
    env.reset()

    sim = env.unwrapped.sim
    model = sim.model

    for i in it.count():

        for j in range(6):

            env.reset()
            model.body_pos[model._body_name2id['robot0:wall_mount'], 2] = 0.9

            for _ in range(40):
                env.render()
                env.step(np.zeros(env.action_space.shape))
                sleep(1 / 60)

            for val in np.linspace(-1, 1, 60):

                env.render()
                action = env.action_space.sample()
                action[-6:] *= 0.0
                action[20+j] = val

                env.step(action)
                sleep(1/60)


if __name__ == '__main__':
    main()
