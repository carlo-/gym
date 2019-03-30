import itertools as it

import gym
from gym.utils.mjviewer import add_selection_logger
from gym.agents.yumi import YumiBarAgent


def main():

    env = gym.make('YumiBar-v1', randomize_initial_object_pos=True)
    raw_env = env.unwrapped
    sim = raw_env.sim
    env.render()
    add_selection_logger(raw_env.viewer, sim)

    agent = YumiBarAgent(env, check_joint_limits=False, use_qp_solver=False)
    done = True
    obs = None

    for _ in it.count():
        if done:
            obs = env.reset()

        u = agent.predict(obs)
        obs, rew, done, _ = env.step(u)
        env.render()
        print(rew)


if __name__ == '__main__':
    main()
