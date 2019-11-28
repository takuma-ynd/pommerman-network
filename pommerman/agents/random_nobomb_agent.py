'''An agent that preforms a random action each step'''
import random
from .. import constants
from . import RandomAgent


class RandomNoBombAgent(RandomAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]
        return random.choice(directions)
