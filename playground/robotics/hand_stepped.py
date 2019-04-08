import itertools as it
from threading import Thread

import numpy as np
import gym
from gym.utils.mjviewer import add_selection_logger, add_scroll_callback

from playground.utils import wait_for_key

selected_action = None
selected_point = None


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


def main():

    env = gym.make(
        'HandStepped-v0',
        render_substeps=True,
    )
    obs = env.reset()

    global selected_action, selected_point
    p = Thread(target=action_thread)
    # p.start()
    selected_action = np.zeros(27)
    selected_point = np.zeros(3)

    def handle_click(selected_pt: np.ndarray, x, y):
        global selected_point
        selected_point[:2] = (x-.5)*2, (y-.5)*2
        print(x, y)

    def handle_scroll(pos):
        global selected_point
        selected_point[2] = pos
        print(pos)

    env.render()
    sim = env.unwrapped.sim_env.sim
    add_selection_logger(env.unwrapped.sim_env.viewer, sim, callback=handle_click)
    add_scroll_callback(env.unwrapped.sim_env.viewer, callback=handle_scroll)
    print('nconmax:', sim.model.nconmax)
    print('obs.shape:', obs['observation'].shape)

    for i in it.count():

        for j in range(6):

            env.reset()
            # for val in np.linspace(-1, 1, 60):
            while True:

                env.render()
                action = np.zeros(15)

                # lfdistal: 3, rfdistal: 2, mfdistal: 1, ffdistal: 0, thdistal: 4

                sel_dir = selected_point / np.linalg.norm(selected_point + 1e-10, ord=2)
                for idx in range(5):
                    action[idx*3:(idx+1)*3] = sel_dir

                closed_pos = 0.5

                # thdistal
                action[12] = -closed_pos
                action[13] = closed_pos

                # ffdistal
                action[0] = closed_pos
                action[1] = closed_pos

                # mfdistal
                action[3] = closed_pos
                action[4] = 0.0

                # lfdistal
                action[9] = -closed_pos
                action[10] = -closed_pos

                # rfdistal
                action[6] = closed_pos
                action[7] = -closed_pos

                rew, done = env.step(action)[1:3]

                # action = selected_action.copy()
                # action[-7:] *= 0.2
                # selected_action[-7:] *= 0.0
                # env.unwrapped.sim_env.step(action)

                print(rew)
                if done:
                    env.reset()


if __name__ == '__main__':
    main()
