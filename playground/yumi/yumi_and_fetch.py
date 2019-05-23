import itertools as it

import numpy as np

import gym
from gym.agents.yumi import YumiConstrainedAgent
from gym.agents.fetch import FetchPushAgent, FetchPickAndPlaceAgent


def main():

    yumi_env = gym.make('YumiConstrainedLong-v2', reward_type='sparse', render_poses=False,
                        has_rotating_platform=False, has_button=True)
    fetch_env = gym.make('FetchPickAndPlaceLong-v1', reward_type='sparse',
                         has_rotating_platform=False, has_button=True)

    yumi_agent = YumiConstrainedAgent(yumi_env)
    # fetch_agent = FetchPushAgent(fetch_env)
    fetch_agent = FetchPickAndPlaceAgent(fetch_env)

    done = True
    yumi_obs = fetch_obs = None
    ep_steps = 0
    yumi_steps_to_success = []
    fetch_steps_to_success = []
    yumi_reached = fetch_reached = False

    for i in it.count():
        if done:
            ep_steps = 0
            yumi_reached = fetch_reached = False
            yumi_obs = yumi_env.reset()
            fetch_obs = fetch_env.reset()

            pos = yumi_env.unwrapped.sim_env._object_xy_pos_to_sync.copy()
            fetch_env.unwrapped.sync_object_init_pos(pos, wrt_table=True, now=False)

            fetch_env.unwrapped.sync_goal(yumi_env.unwrapped.goal, wrt_table=True)

            if len(yumi_steps_to_success) > 0:
                arr = np.asarray(yumi_steps_to_success)
                print('YUMI ===> ', 'min', arr.min(), 'max', arr.max(), 'avg', arr.mean(), 'std', arr.std())
            if len(fetch_steps_to_success) > 0:
                arr = np.asarray(fetch_steps_to_success)
                print('FETCH ==> ', 'min', arr.min(), 'max', arr.max(), 'avg', arr.mean(), 'std', arr.std())

        ep_steps += 1

        u = yumi_agent.predict(yumi_obs)
        yumi_obs, _, done1, info1 = yumi_env.step(u)
        yumi_success = info1['is_success'] == 1

        if yumi_success and not yumi_reached:
            yumi_reached = True
            yumi_steps_to_success.append(ep_steps)

        u = fetch_agent.predict(fetch_obs)
        fetch_obs, _, done2, info2 = fetch_env.step(u)
        fetch_success = info2['is_success'] == 1

        if fetch_success and not fetch_reached:
            fetch_reached = True
            fetch_steps_to_success.append(ep_steps)

        done = done1 and done2
        # yumi_env.render()
        # fetch_env.render()


if __name__ == '__main__':
    main()
