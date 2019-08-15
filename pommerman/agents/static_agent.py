from . import BaseAgent
from .. import characters

class StaticAgent(BaseAgent):
    """just stay calm"""

    def __init__(self, character=characters.Bomber):
        super(StaticAgent, self).__init__(character)

    def act(self, obs, action_space):
        act = 0
        return act
