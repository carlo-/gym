import copy
import pickle
import itertools as it
from threading import Thread

import numpy as np
import gym
from gym.utils.mjviewer import add_selection_logger

from playground.utils import wait_for_key
from gym.agents.shadow_hand import HandPickAndPlaceAgent

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
            print('Object not on the table!')
            while True:
                env.render()
    print('test_pick_and_place PASSED')


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
                print('Arm could not reach the target!')
                while True:
                    env.render()
            obs = env.reset()
    print('test_arm_bounds PASSED')


def generate_grasp_state(max_states=20, file_path=None, render=False):

    env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=False
    )

    agent = HandPickAndPlaceAgent(env)
    obs, hand_ctrl, grasp_steps, env_steps, success_steps = (None,)*5
    reset = True
    found_states = []
    max_env_steps = env.spec.max_episode_steps

    while len(found_states) < max_states:

        if reset:
            obs = env.reset()
            success_steps = 0
            env_steps = 0
            reset = False

        if obs['desired_goal'][2] < 0.48:
            reset = True
            continue

        action = agent.predict(obs)
        obs, reward, _, _ = env.step(action)
        if render:
            env.render()
        if reward == 0.0:
            success_steps += 1
            if success_steps >= 20:
                state = copy.deepcopy(env.unwrapped.sim.get_state())
                found_states.append(state)
                print(f'Found {len(found_states)} so far.')
                reset = True
        else:
            env_steps += 1
            success_steps = 0
        reset = reset or env_steps >= max_env_steps
    print(f'Found {len(found_states)} possible states.')

    stable_states = []
    sim = env.unwrapped.sim
    for state in found_states:
        sim.set_state(copy.deepcopy(state))
        sim.forward()
        stable = True
        for _ in range(200):
            sim.step()
            obj_pos = env.unwrapped._get_object_pose()[:3]
            palm_pos = env.unwrapped._get_palm_pose()[:3]
            if np.linalg.norm(obj_pos - palm_pos) > 0.05:
                stable = False
                break
        if stable:
            print(f'Stable state at index {len(stable_states)}')
            for _ in range(50):
                env.render()
                sim.step()
            stable_states.append(state)

    if len(stable_states) > 0:
        print('Select stable state: ')
        sel = int(input())
        state = stable_states[sel]
        if file_path is not None:
            pickle.dump(state, open(file_path, 'wb'))
        return state

    print('No stable grasps found!')
    return None


def render_reset_states():
    env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=True,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        grasp_state=True,
        grasp_state_reset_p=0.5
    )
    on_palm_count = 0
    for i in it.count():
        env.reset()
        env.render()
        object_pos = env.unwrapped._get_object_pose()[:3]
        palm_pos = env.unwrapped._get_palm_pose(no_rot=True)[:3]
        object_on_palm = np.linalg.norm(object_pos - palm_pos) < 0.08
        on_palm_count += int(object_on_palm)
        print(f'On palm p: {on_palm_count/(i+1)}')
        env.step(env.action_space.sample()*0.0)
        env.render()


def main():
    # env = gym.make('HandPickAndPlaceDense-v0')
    env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=True,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        grasp_state=True,
        grasp_state_reset_p=0.5
    )
    obs = env.reset()

    env.render()
    sim = env.unwrapped.sim
    add_selection_logger(env.unwrapped.viewer, sim)
    print('nconmax:', sim.model.nconmax)
    print('obs.shape:', obs['observation'].shape)

    global selected_action
    p = Thread(target=action_thread)
    # p.start()
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
                # print(rew)
                if done:
                    env.reset()


if __name__ == '__main__':
    main()
