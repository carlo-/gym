from threading import Thread
from time import sleep
import itertools as it

import numpy as np
import mujoco_py
import gym
from gym.envs.robotics import FetchEnv


selected_action = np.zeros(4)


def wait_for_key():
    # From: https://stackoverflow.com/a/34956791

    import sys
    import termios

    result = None
    fd = sys.stdin.fileno()

    old_term = termios.tcgetattr(fd)
    new_attr = termios.tcgetattr(fd)
    new_attr[3] = new_attr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_attr)

    try:
        result = sys.stdin.read(1)
    except IOError:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, old_term)

    return result


def action_from_key():
    key = wait_for_key()
    if key == 'A':
        return np.r_[-1, 0, 0, 0]
    elif key == 'B':
        return np.r_[1, 0, 0, 0]
    elif key == 'C':
        return np.r_[0, 1, 0, 0]
    elif key == 'D':
        return np.r_[0, -1, 0, 0]
    elif key == 'w':
        return np.r_[0, 0, 1, 0]
    elif key == 's':
        return np.r_[0, 0, -1, 0]
    elif key == 'z':
        return np.r_[0, 0, 0, 1]
    elif key == 'x':
        return np.r_[0, 0, 0, -1]
    else:
        return np.zeros(4)


def action_thread():
    global selected_action
    while True:
        a = action_from_key().astype(np.float64)
        if a[-1] == 0:
            selected_action[:3] = a[:3]
        else:
            selected_action = a


def main():
    env = gym.make('FetchPickAndPlaceDense-v1')
    raw_env = env.unwrapped # type: FetchEnv
    raw_env.simplified_reward = True
    sim = raw_env.sim

    global selected_action
    p = Thread(target=action_thread)
    p.start()
    selected_action = np.zeros(4)

    while True:

        env.reset()

        for _ in it.count():

            env.render()
            action = selected_action.copy()
            obs, rew, done, info = env.step(action)

            selected_action[:3] *= 0.0
            sleep(1/60)

            print(rew)
            print(raw_env.get_object_contact_points())


if __name__ == '__main__':
    main()
