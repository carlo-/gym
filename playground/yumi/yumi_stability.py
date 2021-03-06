import gym
from tqdm import tqdm

from playground.utils import test_env_fps


ENV = 'YumiReachTwoArms-v1'
# ENV = 'YumiReachRightArm-v1'
# ENV = 'YumiReachLeftArm-v1'


def test_stability(seed=42):

    env = gym.make(ENV)
    env.seed(seed)
    env.action_space.seed(seed)

    # env.reset()
    # sim = env.unwrapped.sim
    # sim.data.qpos[:] = 0.0
    # sim.data.qvel[:] = 0.0
    # sim.step()

    done = True
    for _ in tqdm(range(500_000)):
        if done:
            env.reset()
        u = env.action_space.sample()
        done = env.step(u)[2]


if __name__ == '__main__':
    test_stability()
    test_env_fps(gym.make('YumiBar-v1'))
    test_env_fps(gym.make('YumiReachTwoArms-v1'))
