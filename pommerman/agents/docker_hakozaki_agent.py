'''An example docker agent.'''
import json
import time
import os
import threading
import requests
import docker
import numpy as np

from . import DockerAgent
from .. import utility
from .. import characters
from .. import constants

def _collapse_alert_by_bombs(observation, ring, remaining_time):
    assert ring > 0
    ring -= 1  # place bombs one ring outer than the actual collapsing ring
    board_shape = observation['board'].shape
    board_size = board_shape[0]

    def place_bombs(r, c):
        '''Handles the collapsing of the board. Will
        kill of remove any item/agent that is on the
        collapsing tile.'''
        if observation['board'][r, c] != constants.Item.Fog.value:
            observation['board'][r, c] = constants.Item.Bomb.value
            observation['bomb_life'][r, c] = remaining_time
            observation['bomb_blast_strength'][r, c] = 2

    for cell in range(ring, board_size - ring):
        place_bombs(ring, cell)
        if ring != cell:
            place_bombs(cell, ring)

        end = board_size - ring - 1
        place_bombs(end, cell)
        if end != cell:
            place_bombs(cell, end)


class DockerHakozakiAgent(DockerAgent):
    """The Docker Agent that Connects to a Docker container where the character runs."""

    def __init__(self, *args, **kwargs):
        super(DockerHakozakiAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        # if collapse will happen soon, replace outer ring with bombs
        print("collapse_ring", obs['collapse_ring'])
        print("collapse_time", obs['collapse_time'])
        if obs['collapse_ring'] > 0 and obs['collapse_time'] > 0:
            ring = obs['collapse_ring']
            time = obs['collapse_time']
            _collapse_alert_by_bombs(obs, ring, time)  # replace outer ring with bombs
        print(obs)

        obs_serialized = json.dumps(obs, cls=utility.PommermanJSONEncoder)
        request_url = "http://localhost:{}/action".format(self._port)
        try:
            req = requests.post(
                request_url,
                # timeout=0.15,
                timeout=1.0,  # temporarily make it longer
                json={
                    "obs":
                    obs_serialized,
                    "action_space":
                    json.dumps(action_space, cls=utility.PommermanJSONEncoder)
                })
            action = req.json()['action']
        except requests.exceptions.Timeout as e:
            print('Timeout!')
            # TODO: Fix this. It's ugly.
            num_actions = len(action_space.shape)
            if num_actions > 1:
                return [0] * num_actions
            else:
                return 0
        return action

