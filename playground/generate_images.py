import gym
from gym.agents.yumi import YumiConstrainedAgent
from gym.agents.fetch import FetchPickAndPlaceAgent
from gym.agents.shadow_hand import HandPickAndPlaceAgent


OUT_DIR = '/home/carlo/KTH/thesis/ms-thesis/figures/'


def _hide_mocap_bodies(viewer, show=False):
    viewer._show_mocap = show
    for body_idx1, val in enumerate(viewer.sim.model.body_mocapid):
        if val != -1:
            for geom_idx, body_idx2 in enumerate(viewer.sim.model.geom_bodyid):
                if body_idx1 == body_idx2:
                    if not show:
                        viewer.sim.extras[geom_idx] = viewer.sim.model.geom_rgba[geom_idx, 3]
                        viewer.sim.model.geom_rgba[geom_idx, 3] = 0
                    else:
                        viewer.sim.model.geom_rgba[geom_idx, 3] = viewer.sim.extras[geom_idx]


def _get_frames(*, env, agent, max_frames=50, seed=44, camera=-1):

    env.seed(seed)
    raw_env = env.unwrapped
    env.render()
    _hide_mocap_bodies(raw_env.viewer)

    done = True
    obs = None
    res = []

    for _ in range(max_frames):
        if done:
            obs = env.reset()
        u = agent.predict(obs)
        obs, rew, done, _ = env.step(u)
        img = env.render(mode='rgb_array', rgb_options=dict(camera_id=camera))
        res.append(img)
    return res


def _yumi_env(**kwargs):
    env = gym.make('YumiConstrained-v2', render_poses=False)
    agent = YumiConstrainedAgent(env)
    return _get_frames(env=env, agent=agent, **kwargs)


def _yumi_env_sphere(**kwargs):
    env = gym.make('YumiConstrained-v2', render_poses=False, object_id='fetch_sphere')
    agent = YumiConstrainedAgent(env)
    return _get_frames(env=env, agent=agent, **kwargs)


def _fetch_env(**kwargs):
    env = gym.make('FetchPickAndPlace-v1')
    agent = FetchPickAndPlaceAgent(env)
    return _get_frames(env=env, agent=agent, **kwargs)


def _fetch_env_button(**kwargs):
    env = gym.make('FetchPickAndPlace-v1', has_button=True)
    agent = FetchPickAndPlaceAgent(env)
    return _get_frames(env=env, agent=agent, **kwargs)


def _fetch_env_platform(**kwargs):
    env = gym.make('FetchPickAndPlace-v1', has_rotating_platform=True)
    agent = FetchPickAndPlaceAgent(env)
    return _get_frames(env=env, agent=agent, **kwargs)


def _fetch_env_sphere(**kwargs):
    env = gym.make('FetchPickAndPlace-v1', object_id='sphere')
    agent = FetchPickAndPlaceAgent(env)
    return _get_frames(env=env, agent=agent, **kwargs)


def _hand_env(**kwargs):
    env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=True,
        randomize_initial_arm_pos=True,
        randomize_initial_object_pos=True,
        object_cage=False,
        object_id='box'
    )
    agent = HandPickAndPlaceAgent(env)
    return _get_frames(env=env, agent=agent, max_frames=100, **kwargs)


def main():

    from PIL import Image

    np_images = dict(
        # yumi=[_yumi_env(), [0, 10, 20, 30, 49]],
        yumi_sphere=[_yumi_env_sphere(), [0, 10, 20, 30, 49]],
        # fetch_front=[_fetch_env(camera=-1), [0, 10, 20, 30, 49]],
        # fetch=[_fetch_env(camera=3), [0, 10, 20, 30, 49]],
        # hand=[_hand_env(camera=-1), [0, 20, 40, 60, 99]],
        # fetch_sphere=[_fetch_env_sphere(camera=3), [0, 10, 20, 30, 49]],
        # fetch_platform=[_fetch_env_platform(camera=-1), [0, 10, 20, 30, 49]],
        # fetch_button=[_fetch_env_button(camera=3), [0, 10, 20, 30, 49]],
    )

    for k, (v, indexes) in np_images.items():
        for i, img_idx in enumerate(indexes):
            np_arr = v[img_idx]
            file_path = f'{OUT_DIR}/frames_{k}_i{i}.png'
            pil_img = Image.fromarray(np_arr)
            pil_img.save(file_path)


if __name__ == '__main__':
    main()
