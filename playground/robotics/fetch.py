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

    while True:

        env.reset()

        for s in it.count():

            env.render()
            action = selected_action.copy()
            obs, rew, done, info = env.step(action)
            print(rew)

            selected_action[:3] *= 0.0
            sleep(1/60)

            if s % 60 != 0:
                continue

            # From: https://gist.github.com/machinaut/209c44e8c55245c0d0f0094693053158
            for i in range(sim.data.ncon):
                # Note that the contact array has more than `ncon` entries,
                # so be careful to only read the valid entries.
                contact = sim.data.contact[i]
                body_name_1 = sim.model.body_id2name(sim.model.geom_bodyid[contact.geom1])
                body_name_2 = sim.model.body_id2name(sim.model.geom_bodyid[contact.geom2])

                if body_name_1 == 'object0' or body_name_2 == 'object0':
                    print('Contact', i)
                    print('geom1', contact.geom1, body_name_1)
                    print('geom2', contact.geom2, body_name_2)

                    geom2_body = sim.model.geom_bodyid[sim.data.contact[i].geom2]
                    print('Total force exerted on geom2 body', sim.data.cfrc_ext[geom2_body])

                    c_force = np.zeros(6, dtype=np.float64)
                    mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, c_force)
                    print('Contact force', c_force)


if __name__ == '__main__':
    main()
