from threading import Thread
from time import sleep
import itertools as it

import numpy as np
import gym
from gym.envs.robotics import FetchEnv
from gym.agents.fetch import FetchPickAndPlaceAgent, FetchPushAgent
from gym.utils.mjviewer import add_selection_logger

from playground.utils import wait_for_key

selected_action = np.zeros(4)


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
    env = gym.make(
        'FetchPickAndPlaceDense-v1',
        reward_params=dict(stepped=True),
        # explicit_goal_distance=True,
        has_rotating_platform=False,
        has_button=False,
        object_id='original',
        has_object_box=True,
    )
    raw_env = env.unwrapped # type: FetchEnv
    sim = raw_env.sim
    env.render()
    add_selection_logger(raw_env.viewer, sim)

    global selected_action
    # p = Thread(target=action_thread)
    # p.start()
    selected_action = np.zeros(4)

    # agent = FetchPickAndPlaceAgent(env)
    agent = FetchPushAgent(env)

    while True:

        obs = env.reset()

        for _ in range(100):

            env.render()
            action = selected_action.copy()
            action = env.action_space.sample()
            action = agent.predict(obs)

            obs, rew, done, info = env.step(action)

            selected_action[:3] *= 0.0
            sleep(1/60)

            print('rew', rew)
            # print(raw_env.get_object_contact_points())


if __name__ == '__main__':
    main()
