from threading import Thread
import itertools as it

import numpy as np
import gym
from gym.envs.yumi.yumi_env import YumiEnv

from playground.utils import wait_for_key


SAMPLE_ACTIONS = True
# ENV = 'YumiTwoArms-v0'
ENV = 'YumiRightArm-v0'


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

    env = gym.make(ENV)
    sim = env.unwrapped.sim
    raw_env = env.unwrapped # type: YumiEnv
    block_gripper = raw_env.block_gripper

    selected_action = np.zeros(env.action_space.shape)
    if not SAMPLE_ACTIONS:
        p = Thread(target=action_thread, args=(selected_action, raw_env.has_two_arms))
        p.start()

    def controller(err, prev_err):
        d_err = (err - prev_err) / raw_env.dt
        return -(1.0 * err + 0.05 * d_err) * 1.0, err.copy()

    env.reset()
    sim.data.qpos[:] = 0.0
    sim.data.qvel[:] = 0.0
    sim.step()

    prev_err1 = np.zeros_like(7)
    prev_err2 = np.zeros_like(7)

    for _ in it.count():
        env.render()

        a = env.action_space.sample()
        if not SAMPLE_ACTIONS:
            a *= 0.0

        a1, prev_err1 = controller(sim.data.qpos[:7], prev_err1)
        a[:7] = a1

        if raw_env.arm == 'both':
            a2, prev_err2 = controller(sim.data.qpos[9:16], prev_err2)
            if block_gripper:
                a[7:] = a2
            else:
                a[8:-1] = a2

        if not block_gripper:
            a += selected_action

        env.step(a)
        selected_action *= 0.0


if __name__ == '__main__':
    main()
