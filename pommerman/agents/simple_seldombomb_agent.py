
import numpy as np

import random
from . import SimpleAgent
from .. import constants

class SimpleSeldomBombAgent(SimpleAgent):
    """
    Basically simple agent but put a bomb only at certain probability
    """

    def __init__(self, bomb_prob=0.25, *args, **kwargs):
        self._bomb_prob = bomb_prob
        assert 0 <= self._bomb_prob and self._bomb_prob <= 1
        super().__init__(*args, **kwargs)

    def act(self, *args, **kwargs):
        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]

        action = super().act(*args, **kwargs)

        # with (1 - bomb_prob) probability, overwrite with a random action.
        if action == constants.Action.Bomb.value:
            if random.random() < (1 - self._bomb_prob):
                action = random.choice(directions)
        return action
