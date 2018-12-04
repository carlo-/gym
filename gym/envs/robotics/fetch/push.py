import os
from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')
SPHERE_MODEL_XML_PATH = os.path.join('fetch', 'push_sphere.xml')


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):

    def __init__(self, reward_type='sparse', custom_xml=None):

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

        fetch_env.FetchEnv.__init__(
            self, xml_path, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


class FetchPushSphereEnv(FetchPushEnv):
    def __init__(self, reward_type='sparse'):
        xml_path = SPHERE_MODEL_XML_PATH
        super().__init__(reward_type=reward_type, custom_xml=xml_path)
