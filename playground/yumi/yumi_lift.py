import itertools as it

import gym
from gym.utils.mjviewer import add_selection_logger
from gym.agents.yumi import YumiLiftAgent


def main():

    env = gym.make('YumiLift-v1', randomize_initial_object_pos=True)
    raw_env = env.unwrapped
    sim = raw_env.sim
    env.render()
    add_selection_logger(raw_env.viewer, sim)

    agent = YumiLiftAgent(env)
    done = True
    obs = None

    for _ in it.count():
        if done:
            agent.reset()
            obs = env.reset()

        u = agent.predict(obs)
        obs, rew, done, _ = env.step(u)
        env.render()
        print(rew)


if __name__ == '__main__':
    main()
