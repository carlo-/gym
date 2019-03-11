import itertools as it

import gym
from gym.agents.yumi import YumiReachAgent
from gym.utils.mjviewer import add_selection_logger


# ENV = 'YumiReachTwoArms-v0'
# ENV = 'YumiReachRightArm-v0'
ENV = 'YumiReachLeftArm-v0'


def main():

    env = gym.make(ENV)
    raw_env = env.unwrapped
    sim = raw_env.sim
    env.render()
    add_selection_logger(raw_env.viewer, sim)

    # env.reset()
    # sim.data.qpos[:] = 0.0
    # sim.data.qvel[:] = 0.0
    # sim.step()

    agent = YumiReachAgent(env)
    done = True
    obs = None

    for _ in it.count():
        if done:
            obs = env.reset()
        u = agent.predict(obs)

        # print(u)
        obs, rew, done, _ = env.step(u)
        env.render()
        print(rew)


if __name__ == '__main__':
    main()
