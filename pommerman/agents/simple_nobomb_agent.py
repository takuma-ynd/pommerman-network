import random
from .. import constants
from . import SimpleAgent


class SimpleNoBombAgent(SimpleAgent):
    """
    Just a SimpleAgent without bomb action
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self, *args, **kwargs):
        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]

        action = super().act(*args, **kwargs)
        if action == constants.Action.Bomb.value:
            action = random.choice(directions)
        return action
