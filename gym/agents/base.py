
class BaseAgent(object):

    def __init__(self, env, **kwargs):
        super(BaseAgent).__init__()
        self._env = env

    def predict(self, obs, **kwargs):
        raise NotImplementedError()


class RandomAgent(BaseAgent):

    def predict(self, obs, **kwargs):
        return self._env.action_space.sample()
