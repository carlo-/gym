import itertools as it

import gym
from gym.agents.yumi import YumiConstrainedAgent
from gym.agents.fetch import FetchPushAgent, FetchPickAndPlaceAgent


def main():

    yumi_env = gym.make('YumiConstrained-v2', reward_type='sparse', render_poses=False,
                        has_rotating_platform=False, has_button=True)
    fetch_env = gym.make('FetchPickAndPlace-v1', reward_type='sparse',
                         has_rotating_platform=False, has_button=True)

    yumi_agent = YumiConstrainedAgent(yumi_env)
    # fetch_agent = FetchPushAgent(fetch_env)
    fetch_agent = FetchPickAndPlaceAgent(fetch_env)

    done = True
    yumi_obs = fetch_obs = None

    for i in it.count():
        if done:
            yumi_obs = yumi_env.reset()
            fetch_obs = fetch_env.reset()

            pos = yumi_env.unwrapped.sim_env._object_xy_pos_to_sync.copy()
            fetch_env.unwrapped.sync_object_init_pos(pos, wrt_table=True, now=False)

            fetch_env.unwrapped.sync_goal(yumi_env.unwrapped.goal, wrt_table=True)

        u = yumi_agent.predict(yumi_obs)
        yumi_obs, _, done1, _ = yumi_env.step(u)

        u = fetch_agent.predict(fetch_obs)
        fetch_obs, _, done2, _ = fetch_env.step(u)

        done = done1 or done2
        yumi_env.render()
        fetch_env.render()


if __name__ == '__main__':
    main()
