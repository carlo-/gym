from gym import utils
from gym.envs.robotics import fetch_env


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):

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
            xml_path = 'fetch/pick_and_place.xml'

        fetch_env.FetchEnv.__init__(
            self, xml_path, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


class FetchPickAndPlaceSphereEnv(FetchPickAndPlaceEnv):
    def __init__(self, reward_type='sparse'):
        xml_path = 'fetch/pick_and_place_sphere.xml'
        super().__init__(reward_type=reward_type, custom_xml=xml_path)
