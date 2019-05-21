import os
from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')
SPHERE_MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place_sphere.xml')


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):

    def __init__(self, custom_xml=None, **kwargs):

        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }

        if custom_xml is not None and isinstance(custom_xml, str):
            xml_path = custom_xml
        else:
            xml_path = MODEL_XML_PATH

        default_kwargs = dict(
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=(0.1, 0.15), target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse', has_rotating_platform=False,
        )

        merged = {**default_kwargs, **kwargs}
        fetch_env.FetchEnv.__init__(self, xml_path, has_object=True, **merged)
        utils.EzPickle.__init__(self)


class FetchPickAndPlaceSphereEnv(FetchPickAndPlaceEnv):
    def __init__(self, **kwargs):
        super().__init__(custom_xml=SPHERE_MODEL_XML_PATH, **kwargs)


class FetchPickAndPlaceEasyEnv(FetchPickAndPlaceEnv):
    def __init__(self, **kwargs):
        default_kwargs = dict(gripper_extra_height=0.15, obj_range=0.0)
        merged = {**default_kwargs, **kwargs}
        super().__init__(**merged)
