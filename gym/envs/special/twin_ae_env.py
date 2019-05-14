import numpy as np

import gym
from gym.agents.base import BaseAgent


class TwinAutoencoderEnv(gym.Env):

    def __init__(self, *, teacher_agent: BaseAgent, teacher_env: gym.Env, student_env: gym.Env, twin_ae_model,
                 student_is_a_env=True, student_obs_transform=None, teacher_obs_transform=None, sync_goals=None,
                 task_rew_weight=1.0, imitation_rew_weight=1.0):

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
        }

        self.action_space = student_env.action_space
        self.observation_space = student_env.observation_space

        self.teacher_agent = teacher_agent
        self.teacher_env = teacher_env.unwrapped
        self.student_env = student_env.unwrapped
        self.twin_ae_model = twin_ae_model
        self.task_rew_weight = task_rew_weight
        self.imitation_rew_weight = imitation_rew_weight

        self._student_is_a_env = student_is_a_env
        self._student_obs_transform = student_obs_transform
        self._teacher_obs_transform = teacher_obs_transform
        self._sync_goals = sync_goals

        self._teacher_env_ep_len = teacher_env.spec.timestep_limit
        self._student_env_ep_len = student_env.spec.timestep_limit

        self._teacher_env_ep_steps = 0
        self._student_env_ep_steps = 0
        self._teacher_student_ep_len_ratio = self._teacher_env_ep_len / self._student_env_ep_len

        self._teacher_last_obs = None
        self._student_last_obs = None

        self.seed()

    def _compute_imitation_reward(self, student_obs, teacher_obs):

        if callable(self._student_obs_transform):
            student_obs = self._student_obs_transform(student_obs)

        if callable(self._teacher_obs_transform):
            teacher_obs = self._teacher_obs_transform(teacher_obs)

        if self._student_is_a_env:
            a_obs = student_obs
            b_obs = teacher_obs
        else:
            b_obs = student_obs
            a_obs = teacher_obs

        sim_loss = self.twin_ae_model.compute_obs_sim_loss(a_obs, b_obs)
        return sim_loss.mean()

    def seed(self, seed=None):
        self.student_env.seed(seed=seed)
        self.teacher_env.seed(seed=seed)

    def reset(self):
        self.teacher_agent.reset()
        t_obs = self.teacher_env.reset()
        s_obs = self.student_env.reset()

        if callable(self._sync_goals):
            res = self._sync_goals(t_env=self.teacher_env, t_obs=t_obs, s_env=self.student_env, s_obs=s_obs)
            t_obs, s_obs = res['t_obs'], res['s_obs']

        self._teacher_last_obs = t_obs
        self._student_last_obs = s_obs

        self._student_env_ep_steps = 0
        self._teacher_env_ep_steps = 0

        return s_obs

    def step(self, action):

        s_obs, s_rew, s_done, s_info = self.student_env.step(action)
        self._student_env_ep_steps += 1
        self._student_last_obs = s_obs

        imitation_reward = -self._compute_imitation_reward(s_obs, self._teacher_last_obs)
        task_reward = s_rew
        tot_reward = imitation_reward * self.imitation_rew_weight + task_reward * self.task_rew_weight

        teacher_des_step = int(self._teacher_student_ep_len_ratio * self._student_env_ep_steps)
        teacher_des_step = min(self._teacher_env_ep_len, teacher_des_step)
        teacher_needed_steps = max(0, teacher_des_step - self._teacher_env_ep_steps)

        for _ in range(teacher_needed_steps):
            teacher_act = self.teacher_agent.predict(self._teacher_last_obs)
            t_obs, t_rew, t_done, t_info = self.teacher_env.step(teacher_act)
            self._teacher_env_ep_steps += 1
            self._teacher_last_obs = t_obs

        if s_done or self._student_env_ep_steps >= self._student_env_ep_len:
            _ = self.reset()

        return s_obs, tot_reward, s_done, s_info

    def render(self, **kwargs):
        s_res = self.student_env.render(**kwargs)
        t_res = self.teacher_env.render(**kwargs)
        return s_res, t_res


def _test_env():

    ms_thesis_path = '/home/carlo/KTH/thesis/ms-thesis'

    import sys
    sys.path.insert(0, '/home/carlo/KTH/thesis/thesis_env/lib/python3.6/site-packages')
    sys.path.insert(0, f'{ms_thesis_path}/playground')
    sys.path.insert(0, ms_thesis_path)

    from twin_vae import TwinVAE, TwinDataset, SimpleAutoencoder
    from gym.agents.shadow_hand import HandPickAndPlaceAgent
    from gym.agents.yumi import YumiConstrainedAgent
    from gym.utils import transformations as tf

    teacher_env = gym.make(
        'HandPickAndPlace-v0',
        ignore_rotation_ctrl=True,
        ignore_target_rotation=True,
        success_on_grasp_only=False,
        randomize_initial_arm_pos=False,
        randomize_initial_object_pos=True,
        object_id='box'
    )
    student_env = gym.make(
        'YumiConstrained-v2'
    )

    teacher_agent = HandPickAndPlaceAgent(teacher_env)
    student_agent = YumiConstrainedAgent(student_env)

    ae_model = TwinVAE.load(
        f'{ms_thesis_path}/out/twin_yumi_hand_ae_resets_fixed_long/checkpoints/model_c1.pt',
        net_class=SimpleAutoencoder
    )
    dataset = TwinDataset.load(f'{ms_thesis_path}/out/pp_yumi_hand_fixed_twin_dataset_3k.pkl')
    dataset.normalize()

    s_table_tf = student_env.unwrapped.get_table_surface_pose()
    t_table_tf = teacher_env.unwrapped.get_table_surface_pose()

    def _student_obs_transformer(obs_dict):
        o = np.r_[obs_dict['observation'], obs_dict['desired_goal']]
        return dataset.a_scaler.transform(o[None], copy=True)[0]

    def _teacher_obs_transformer(obs_dict):
        o = np.r_[obs_dict['observation'], obs_dict['desired_goal']]
        return dataset.b_scaler.transform(o[None], copy=True)[0]

    def _sync_goals(*, t_env, s_env, **kwargs_):
        tf_to_goal = tf.get_tf(np.r_[s_env.goal, 1., 0., 0., 0.], s_table_tf)
        t_goal_pose = tf.apply_tf(tf_to_goal, t_table_tf)

        t_env.goal = np.r_[t_goal_pose[:3], np.zeros(4)]

        tf_to_obj = tf.get_tf(s_env.get_object_pose(), s_table_tf)
        t_obj_pose = tf.apply_tf(tf_to_obj, t_table_tf)

        object_pos = t_env.sim.data.get_joint_qpos('object:joint').copy()
        object_pos[:2] = t_obj_pose[:2]
        t_env.sim.data.set_joint_qpos('object:joint', object_pos)
        t_env.sim.forward()

        return dict(s_obs=s_env._get_obs(), t_obs=t_env._get_obs())

    kwargs = dict(
        teacher_agent=teacher_agent,
        teacher_env=teacher_env,
        student_env=student_env,
        twin_ae_model=ae_model,
        student_is_a_env=True,
        student_obs_transform=_student_obs_transformer,
        teacher_obs_transform=_teacher_obs_transformer,
        sync_goals=_sync_goals,
    )

    env = gym.make('TwinAutoencoder-v0', **kwargs)
    done = True
    obs = None

    # from playground.utils import test_env_fps
    # test_env_fps(env, steps=1000)

    while True:
        if done:
            obs = env.reset()
        # action = env.action_space.sample()
        action = student_agent.predict(obs)
        obs, rew, done, info = env.step(action)
        env.render()
        print(rew)


if __name__ == '__main__':
    _test_env()
