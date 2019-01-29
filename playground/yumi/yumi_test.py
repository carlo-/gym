from threading import Thread
from time import sleep
from datetime import datetime
import itertools as it

import numpy as np
import gym
from gym.envs.yumi.yumi_env import YumiEnv

from playground.utils import wait_for_key


def action_thread(selected_action, has_two_arms):

    def safe_int(x):
        try:
            return int(x)
        except ValueError:
            return None

    s = 1.0
    while True:
        key = wait_for_key()
        print(key)
        a = np.zeros_like(selected_action)
        i = safe_int(key)

        if i is not None:
            a[7] = 0.1 * i * s
        elif key == 'A':
            a[7] = 1.0 * s
        elif key == 'B':
            a[7] = -1.0 * s
        elif key == 'w' and has_two_arms:
            a[7+8] = 1.0 * s
        elif key == 's' and has_two_arms:
            a[7+8] = -1.0 * s
        elif key == 'm':
            s *= -1.0
        selected_action[:] = a


def main():

    env = gym.make('YumiRightArm-v0')
    env.reset()
    raw_env = env.unwrapped # type: YumiEnv

    selected_action = np.zeros(env.action_space.shape)
    p = Thread(target=action_thread, args=(selected_action, raw_env.has_two_arms))
    p.start()

    sim = env.unwrapped.sim
    model = sim.model

    def controller(err, prev_err):
        d_err = (err - prev_err) / raw_env.dt
        return -(1.0 * err + 0.05 * d_err) * 1.0, err.copy()

    for i in it.count():

        env.reset()
        sim.data.qpos[:] = 0.0
        sim.data.qvel[:] = 0.0
        sim.step()

        a = np.zeros(env.action_space.shape)
        prev_err1 = np.zeros_like(7)
        prev_err2 = np.zeros_like(7)

        for j in range(2000000):
            env.render()

            a *= 0.0

            a1, prev_err1 = controller(sim.data.qpos[:7], prev_err1)
            a[:7] = a1

            if raw_env.arm == 'both':
                a2, prev_err2 = controller(sim.data.qpos[9:16], prev_err2)
                a[8:-1] = a2

            a += selected_action

            env.step(a)
            selected_action *= 0.0


if __name__ == '__main__':
    main()
