import sys

import gym
from gym.agents.fetch import FetchPickAndPlaceAgent, FetchPushAgent
from gym.agents.yumi import YumiImitatorAgent


MS_THESIS_PATH = '/home/carlo/KTH/thesis/ms-thesis'
sys.path.insert(0, '/home/carlo/KTH/thesis/thesis_env/lib/python3.6/site-packages')
sys.path.insert(0, f'{MS_THESIS_PATH}/playground')
sys.path.insert(0, MS_THESIS_PATH)

from twin_vae import TwinVAE, TwinDataset, SimpleAutoencoder


def from_fetch():
    # model = TwinVAE.load('../out/twin_yumi_v2_fetch_ae_resets_new_l/checkpoints/model_c1.pt',
    #                      net_class=SimpleAutoencoder)
    # dataset = TwinDataset.load('../out/pp_yumi_v2_fetch_twin_dataset_10k.pkl')
    model = TwinVAE.load(f'{MS_THESIS_PATH}/out/twin_yumi_v2_fetch_ae_resets_ext_bounds2/checkpoints/model_c1.pt',
                         net_class=SimpleAutoencoder)
    dataset = TwinDataset.load(f'{MS_THESIS_PATH}/out/pp_yumi_v2_fetch_ext_bounds_twin_dataset_15k.pkl')
    dataset.normalize()

    teacher_env = gym.make(
        'FetchPickAndPlaceLong-v1',
        reward_type='sparse',
        has_rotating_platform=False,
        object_id='sphere',
    )

    student_env = gym.make(
        'YumiConstrainedLong-v2',
        reward_type='sparse',
        render_poses=False,
        object_on_table=True,
        has_rotating_platform=False,
        has_button=False,
        object_id='fetch_sphere',
    )

    teacher = FetchPickAndPlaceAgent(teacher_env)
    # teacher = FetchPushAgent(teacher_env)
    agent = YumiImitatorAgent(student_env, teacher_env=teacher_env, teacher_agent=teacher, a_scaler=dataset.a_scaler,
                              b_scaler=dataset.b_scaler, model=model)

    done = True
    obs = None

    while True:
        if done:
            obs = student_env.reset()
            agent.reset()

        u = agent.predict(obs)
        obs, rew, done, info = student_env.step(u)
        #
        # if done:
            # if info['is_success'] == 1:
            #     print('yey')
            # else:
            #     print('ney')

        student_env.render()
        teacher_env.render()


if __name__ == '__main__':
    from_fetch()
