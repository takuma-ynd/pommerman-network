"""The Pommerman v1.6 Environment. Designed for RL agents.
Reward shaping is implemented.

This environment is the same as v0.py, except that the board will collapse
according to a uniform schedule beginning at the kwarg first_collapse.

The collapsing works in the following manner:
1. Set the collapsing schedule. This is uniform starting at step first_collapse
   and ending at step max_steps.
2. Number the rings on the board from 0 to board_size-1 s.t. the outermost ring
   is 0 and the innermost ring is board_size-1. The cells in the former are
   [[0, i], [i, 0], [board_size-1, i], [i, board_size-1] for i in
   [0, board_size-1]] and the latter is, assuming an odd board_size,
   [[(board_size-1)/2, (board_size-1)/2]].
3. When we are at a step in the collapsing schedule, we take the matching ring
   and turn it into rigid walls. This has the effect of destroying any items,
   bombs (which don't go off), and agents in those squares.
"""
import time
import json
from .. import constants
from .. import utility
from .. import graphics
from . import v1_5
import numpy as np


class AbilityTracker:
    def __init__(self, agents):
        self.abilities = []
        for agent in agents:
            ability = {key: getattr(agent, key) for key in ['ammo_capacity', 'blast_strength', 'can_kick']}
            self.abilities.append(ability)

    def diff(self, another_ability):
        ret = []
        assert len(self.abilities) == len(another_ability)
        for i in range(len(self.abilities)):
            assert len(self.abilities[i]) == len(another_ability[i])
            _dict = {}
            for key in self.abilities[i]:
                _dict[key] =  self.abilities[i][key] - another_ability[i][key]
            ret.append(_dict)

        # ret[1]['ammo'] = self.abilities[1]['ammo'] - another_ability[1]['ammo']
        return ret

    def __getitem__(self, idx):
        return self.abilities[idx]

    def __len__(self):
        return len(self.abilities)




class Pomme(v1_5.Pomme):
    '''version 1.6: this just adds a reward shaping on top of version 1.5.
    This is designed for RL agent training.
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array', 'rgb_pixel'],
        'video.frames_per_second': constants.RENDER_FPS
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previous_ability = None
        self._prev_is_alive = [True for _ in range(4)]


    def reset(self):
        observations = super().reset()
        self._previous_ability = None
        self._prev_is_alive = [True for _ in range(4)]
        return observations

    def _get_rewards(self):
        '''when ability of an agent increases, it gets reward'''
        reward = super()._get_rewards()
        return reward

    def _get_done(self):
        done = self.model.get_done(self._agents, self._step_count,
                                   self._max_steps, self._game_type,
                                   self.training_agent)
        return done
