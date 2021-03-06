import itertools as it

import numpy as np

import gym
from gym.utils.mjviewer import add_selection_logger
from gym.utils.transformations import render_pose
from gym.agents.yumi import YumiConstrainedAgent
from playground.utils import test_env_fps


def test_fps(steps=10_000):
    test_env_fps(gym.make('YumiConstrained-v2'), steps)
    test_env_fps(gym.make('YumiConstrained-v1'), steps)


def test_reachability(render=False):

    env = gym.make('YumiConstrainedLong-v2', reward_type='sparse', render_poses=False,
                   has_rotating_platform=False, has_button=False, extended_bounds=True)

    if render:
        raw_env = env.unwrapped
        sim = raw_env.sim
        env.render()
        add_selection_logger(raw_env.viewer, sim)

    agent = YumiConstrainedAgent(env)

    goal_reach = np.zeros((2, 3))
    goal_reach[0] = np.inf
    goal_reach[1] = -np.inf
    init_reach = goal_reach.copy()

    done = True
    obs = init_pos = None
    n_steps = 0
    unreachable_eps = 0
    tot_eps = 0

    for i in it.count():
        if done:
            obs = env.reset()
            init_pos = obs['achieved_goal'].copy()
            n_steps = 0
            tot_eps += 1

        u = agent.predict(obs)

        obs, rew, done, _ = env.step(u)
        n_steps += 1

        if render:
            env.render()

        if rew == 0.0:

            done = True
            if n_steps >= 10:

                goal = obs['desired_goal'].copy()
                goal_reach[0] = np.minimum(goal_reach[0], goal)
                goal_reach[1] = np.maximum(goal_reach[1], goal)

                init_reach[0] = np.minimum(init_reach[0], init_pos)
                init_reach[1] = np.maximum(init_reach[1], init_pos)

        elif done:
            unreachable_eps += 1

        if done:
            print(goal_reach)
            print(init_reach)
            print(unreachable_eps / tot_eps)
            print()


def main():

    env = gym.make('YumiConstrained-v2', reward_type='sparse', render_poses=False,
                   has_rotating_platform=False, has_button=False, has_object_box=False, object_id="fetch_sphere")
    raw_env = env.unwrapped
    sim = raw_env.sim
    env.render()
    add_selection_logger(raw_env.viewer, sim)

    agent = YumiConstrainedAgent(env)

    done = True
    obs = None
    steps_to_success = []
    n_steps = 0

    reachability = np.zeros((2, 3))
    reachability[0] = np.inf
    reachability[1] = -np.inf
    unreachable_eps = 0
    tot_eps = 0

    for i in it.count():
        if done:
            obs = env.reset()
            n_steps = 0
            tot_eps += 1

        # center_pos = obs['observation'][:3]
        # achieved_goal = obs['achieved_goal']
        # err = achieved_goal - center_pos
        # # u = np.zeros(env.action_space.shape)
        # # u[1:4] = err * 10.0

        u = env.action_space.sample()
        u[0] = -1 if i // 8 % 2 == 0 else 1
        u[1:] = 0.

        u = agent.predict(obs)

        render_pose(env.unwrapped.get_table_surface_pose(), env.unwrapped.viewer, unique_id=535)

        obs, rew, done, _ = env.step(u)
        # done = False
        n_steps += 1
        env.render()

        if rew == 0.0:
            steps_to_success.append(n_steps)
            arr = np.asarray(steps_to_success)
            # print('min', arr.min(), 'max', arr.max(), 'avg', arr.mean(), 'std', arr.std())
            done = True

            goal = obs['desired_goal'].copy()
            reachability[0] = np.minimum(reachability[0], goal)
            reachability[1] = np.maximum(reachability[1], goal)

            # print(reachability)
            # print(unreachable_eps / tot_eps)
            # print()

        elif done:
            unreachable_eps += 1


if __name__ == '__main__':
    main()
