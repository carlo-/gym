import itertools as it
from threading import Thread

import numpy as np
import gym
from gym.utils.mjviewer import add_selection_logger

from playground.utils import wait_for_key

selected_action = None


def action_from_key():
    key = wait_for_key()
    pos = np.zeros(3)
    rot = np.zeros(4)
    wrist_ctrl = np.zeros(2)
    hand_ctrl = np.zeros(18)
    if key == 'A':
        pos = np.r_[-1, 0, 0]
    elif key == 'B':
        pos = np.r_[1, 0, 0]
    elif key == 'C':
        pos = np.r_[0, 1, 0]
    elif key == 'D':
        pos = np.r_[0, -1, 0]
    elif key == 'w':
        pos = np.r_[0, 0, 1]
    elif key == 's':
        pos = np.r_[0, 0, -1]
    elif key == 'c':
        hand_ctrl[:] = 1.0
        hand_ctrl[13:] = 0.0
    elif key == 'o':
        hand_ctrl[:] = -1.0
        hand_ctrl[13:] = 0.0
    elif key == 't':
        hand_ctrl[13:] = (1., 1., 1., -1., -1.)
    elif key == 'y':
        hand_ctrl[13:] = (-1., 1., 1., -1., -1.)
    elif key == 'x':
        wrist_ctrl[1] = 1.0
    elif key == 'z':
        wrist_ctrl[1] = -1.0
    return np.r_[wrist_ctrl*0.2, hand_ctrl*0.2, pos, rot]


def action_thread():
    global selected_action
    while True:
        a = action_from_key().astype(np.float64)
        a_max = np.ones(a.shape)
        a_min = -a_max
        # a_max[15] = 0.1
        selected_action = np.clip(selected_action + a, a_min, a_max)


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


def test_arm_bounds():
    from tqdm import tqdm
    env = gym.make('HandPickAndPlace-v0')
    obs = env.reset()
    for _ in tqdm(range(50_000)):
        action = np.zeros(env.action_space.shape)
        d = (obs['desired_goal'] - env.unwrapped._get_palm_pose(no_rot=True))[:3]
        action[-7:-4] = d * 2.0
        reached = np.linalg.norm(d) < 0.01
        obs, _, done, _ = env.step(action)
        if done or reached:
            if not reached:
                raise RuntimeError('Arm could not reach the target!')
            obs = env.reset()


def main():
    # env = gym.make('HandPickAndPlaceDense-v0')
    env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=True,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=False
    )
    env.reset()

    env.render()
    sim = env.unwrapped.sim
    add_selection_logger(env.unwrapped.viewer, sim)
    print('nconmax:', sim.model.nconmax)

    global selected_action
    p = Thread(target=action_thread)
    p.start()
    selected_action = np.zeros(27)

    for i in it.count():

        for j in range(6):

            env.reset()
            # for val in np.linspace(-1, 1, 60):
            while True:

                env.render()
                action = selected_action.copy()
                action[-7:] *= 0.2
                selected_action[-7:] *= 0.0

                rew, done = env.step(action)[1:3]
                print(rew)
                # if done:
                #     env.reset()


if __name__ == '__main__':
    main()
