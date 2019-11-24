'''This is the base abstraction for agents in pommerman.
All agents should inherent from this class'''
from .. import characters
from copy import deepcopy


class BaseAgent:
    """Parent abstract Agent."""

    def __init__(self, character=characters.Bomber):
        self._character = character

    def __getattr__(self, attr):
        # HACK: to avoid infinite recursive copy error in deepcopy
        # without this if sentence, deepcopy tries to find self._character that doesn't exist, and loops forever
        # if '_character' not in dir(self):  <-- This was significantly slower!!!
        if attr == '_character':
            self._character = characters.Bomber
        return getattr(self._character, attr)

    def act(self, obs, action_space):
        raise NotImplementedError()

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.

        Args:
          reward: The single reward scalar to this agent.
        """
        pass

    def init_agent(self, id_, game_type, env_info=None):
        self._character = self._character(id_, game_type)

    @staticmethod
    def has_user_input():
        return False

    def shutdown(self):
        pass
