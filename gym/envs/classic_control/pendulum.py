import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import scipy.stats
from typing import Sequence


class PendulumEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.viewer = None
        self.np_random = None
        self.last_u = None
        self.state = None

        self.seed()

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.gravity_distr = lambda: 10.
        self.mass_distr = lambda: 1.
        self.length_distr = lambda: 1.

        self.sampled_mass = None
        self.sampled_gravity = None
        self.sampled_length = None
        self.sample_physical_props_at_step = False
        self.length_scale = 1.0

    @property
    def physical_props(self):
        return self.sampled_gravity, self.sampled_mass, self.sampled_length

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        th, thdot = self.state
        g, m, l = self.physical_props
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])

        if self.sample_physical_props_at_step:
            self._change_physical_props()

        return self._get_obs(), -costs, False, {}

    def _change_physical_props(self):
        self.sampled_gravity = max(self.gravity_distr(), 0.001)
        self.sampled_mass = max(self.mass_distr(), 0.001)
        self.sampled_length = max(self.length_distr(), 0.001)

    def reset(self):
        self._change_physical_props()
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        self.pole_transform.set_scale(self.length_scale, 1.0)

        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()


class GaussianPendulumEnv(PendulumEnv):

    def __init__(self):
        super().__init__()
        self.mass_distr_params = None
        self.mass_mean_ranges = None
        self.mass_stdev_ranges = None
        self.embed_knowledge = False
        self.perfect_knowledge = False
        self.configure()

    def configure(self, seed=None, mass_mean=1.0, mass_stdev=0.0, embed_knowledge=False, sample_props_at_step=False,
                  perfect_knowledge=False, gym_env=None, **kwargs):

        if perfect_knowledge:
            assert embed_knowledge, 'Model cannot have perfect knowledge without embedding additional observations!'

        self.seed(seed)

        mass_mean = to_multimodal(mass_mean)
        mass_stdev = to_multimodal(mass_stdev)

        self.mass_mean_ranges = mass_mean
        self.mass_stdev_ranges = mass_stdev
        self.mass_distr_params = None
        self.embed_knowledge = embed_knowledge
        self.sample_physical_props_at_step = sample_props_at_step
        self.perfect_knowledge = perfect_knowledge

        low = self.observation_space.low[:3]
        high = self.observation_space.high[:3]

        if embed_knowledge:
            low = np.concatenate((low, [np.min(mass_mean), np.min(mass_stdev)]))
            high = np.concatenate((high, [np.max(mass_mean), np.max(mass_stdev)]))
            if perfect_knowledge:
                low = low[:4]
                high = high[:4]

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if gym_env is not None and \
                hasattr(gym_env, 'observation_space') and \
                isinstance(gym_env.observation_space, spaces.Box):
            gym_env.observation_space = self.observation_space

    def _change_distribution(self):

        np_random = self.np_random

        mass_mean_range_i = np_random.randint(0, len(self.mass_mean_ranges))
        mass_mean_range = self.mass_mean_ranges[mass_mean_range_i]

        mass_stdev_range_i = np_random.randint(0, len(self.mass_stdev_ranges))
        mass_stdev_range = self.mass_stdev_ranges[mass_stdev_range_i]

        mass_mean = np_random.uniform(*mass_mean_range)
        mass_stdev = 0.0
        for _ in range(5):
            mass_stdev = np_random.uniform(*mass_stdev_range)
            if mass_stdev == 0.0:
                break
            p = scipy.stats.norm.cdf(0.0, loc=mass_mean, scale=mass_stdev)
            if p < 0.05:
                break
        self.mass_distr_params = np.array((mass_mean, mass_stdev))
        self.mass_distr = lambda: np_random.normal(mass_mean, mass_stdev)

    def reset(self):
        self._change_distribution()
        return super().reset()

    def _get_obs(self):
        obs = super()._get_obs()
        if self.embed_knowledge:
            if self.perfect_knowledge:
                exact_mass = self.physical_props[1]
                obs = np.concatenate((obs, [exact_mass]))
            else:
                obs = np.concatenate((obs, self.mass_distr_params))
        return obs


def angle_normalize(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi


def to_multimodal(x):

    def check_val(z):
        return isinstance(z, float) or isinstance(z, int)

    def convert_range(z):
        assert len(z) == 2
        assert check_val(z[0]) and check_val(z[1])
        assert z[0] <= z[1]
        z[0] = float(z[0])
        z[1] = float(z[1])

    if check_val(x):
        return [[float(x), float(x)]]

    if isinstance(x, Sequence):
        assert len(x) > 0
        if isinstance(x[0], Sequence):
            x = list(x)
            for i, y in enumerate(x):
                y = list(y)
                convert_range(y)
                x[i] = y
            return x
        else:
            x = list(x)
            convert_range(x)
            return [x]
    else:
        raise ValueError()
