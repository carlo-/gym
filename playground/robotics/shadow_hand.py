import itertools as it
from threading import Thread

import numpy as np
import gym
from gym.utils.mjviewer import add_selection_logger

from playground.utils import wait_for_key

selected_action = np.zeros(4)


def action_from_key():
    key = wait_for_key()
    if key == 'A':
        return np.r_[-1, 0, 0, 0, 0, 0, 0]
    elif key == 'B':
        return np.r_[1, 0, 0, 0, 0, 0, 0]
    elif key == 'C':
        return np.r_[0, 1, 0, 0, 0, 0, 0]
    elif key == 'D':
        return np.r_[0, -1, 0, 0, 0, 0, 0]
    elif key == 'w':
        return np.r_[0, 0, 1, 0, 0, 0, 0]
    elif key == 's':
        return np.r_[0, 0, -1, 0, 0, 0, 0]
    elif key == 'z':
        return np.r_[0, 0, 0, 1, 0, 0, 0]
    elif key == 'x':
        return np.r_[0, 0, 0, -1, 0, 0, 0]
    else:
        return np.zeros(7)


def action_thread():
    global selected_action
    while True:
        a = action_from_key().astype(np.float64)
        selected_action = a


def test_pick_and_place():
    from tqdm import tqdm
    env = gym.make('HandPickAndPlace-v0')
    sim = env.unwrapped.sim
    done = True
    for _ in tqdm(range(50_000)):
        if done:
            env.reset()
        action = np.zeros(env.action_space.shape)
        done = env.step(action)[2]
        obj_pos = sim.data.get_body_xpos('object')
        if obj_pos[2] < 0.42:
            raise RuntimeError('Object not on the table!')


def main():
    # env = gym.make('HandPickAndPlaceDense-v0')
    env = gym.make('HandPickAndPlace-v0', ignore_rotation_ctrl=True, ignore_target_rotation=True, success_on_grasp_only=True)
    env.reset()

    env.render()
    sim = env.unwrapped.sim
    add_selection_logger(env.unwrapped.viewer, sim)
    print('nconmax:', sim.model.nconmax)

    global selected_action
    p = Thread(target=action_thread)
    p.start()
    selected_action = np.zeros(7)

    for i in it.count():

        for j in range(6):

            env.reset()
            # for val in np.linspace(-1, 1, 60):
            while True:

                env.render()
                action = env.action_space.sample()
                # action[-7:] *= 0.0
                # action[20+j] = val
                action[-7:] = selected_action * 0.2
                selected_action *= 0.0

                rew = env.step(action)[1]
                print(rew)


if __name__ == '__main__':
    main()
