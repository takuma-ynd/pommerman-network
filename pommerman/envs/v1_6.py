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
from . import v0
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




class Pomme(v0.Pomme):
    '''version 1.6: this just adds a reward shaping on top of version 1.5.
    This is designed for RL agent training.
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array', 'rgb_pixel'],
        'video.frames_per_second': constants.RENDER_FPS
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        first_collapse = kwargs.get('first_collapse')
        # collapses is [50, range(fist, max, int((max-first)/3))]
        # ex: first = 100, max = 400 --> [50, 100, 200, 300]
        if first_collapse <= 50:
            first_collapse = 100
        self.collapses = [50] + list(
            range(first_collapse, self._max_steps,
                  int((self._max_steps - first_collapse) / 3)))

        # self.collapses = list(
        #     range(first_collapse, self._max_steps,
        #           int((self._max_steps - first_collapse) / 4)))
        self._collapse_alert_map = np.zeros((self._board_size, self._board_size))
        self._collapse_ring = -1
        self._collapse_time = -1
        self._previous_ability = None
        self._prev_is_alive = [True for _ in range(4)]

    def _collapse_board(self, ring):
        """Collapses the board at a certain ring radius.

        For example, if the board is 13x13 and ring is 0, then the the ring of
        the first row, last row, first column, and last column is all going to
        be turned into rigid walls. All agents in that ring die and all bombs
        are removed without detonating.
        
        For further rings, the values get closer to the center.

        Args:
          ring: Integer value of which cells to collapse.
        """
        board = self._board.copy()

        def collapse(r, c):
            '''Handles the collapsing of the board. Will
            kill of remove any item/agent that is on the
            collapsing tile.'''
            if utility.position_is_agent(board, (r, c)):
                # Agent. Kill it.
                num_agent = board[r][c] - constants.Item.Agent0.value
                agent = self._agents[num_agent]
                agent.die()
            if utility.position_is_bomb(self._bombs, (r, c)):
                # Bomb. Remove the bomb. Update agent's ammo tally.
                new_bombs = []
                for b in self._bombs:
                    if b.position == (r, c):
                        b.bomber.incr_ammo()
                    else:
                        new_bombs.append(b)
                self._bombs = new_bombs
            if utility.position_is_flames(board, (r, c)):
                self._flames = [f for f in self._flames if f.position != (r,c)]
            if (r, c) in self._items:
                # Item. Remove the item.
                del self._items[(r, c)]
            board[r][c] = constants.Item.Rigid.value

        for cell in range(ring, self._board_size - ring):
            collapse(ring, cell)
            if ring != cell:
                collapse(cell, ring)

            end = self._board_size - ring - 1
            collapse(end, cell)
            if end != cell:
                collapse(cell, end)

        return board

    def _alert_collapse(self, ring, remaining_time):
        alert_map = np.zeros(self._board.shape)

        def alert_collapse(r, c):
            '''Handles the collapsing of the board. Will
            kill of remove any item/agent that is on the
            collapsing tile.'''
            alert_map[r][c] = remaining_time

        for cell in range(ring, self._board_size - ring):
            alert_collapse(ring, cell)
            if ring != cell:
                alert_collapse(cell, ring)

            end = self._board_size - ring - 1
            alert_collapse(end, cell)
            if end != cell:
                alert_collapse(cell, end)

        return alert_map


    def get_json_info(self):
        ret = super().get_json_info()
        ret['collapses'] = json.dumps(self.collapses, cls=utility.PommermanJSONEncoder)
        return ret

    def set_json_info(self):
        super().set_json_info()
        self.collapses = json.loads(self._init_game_state['collapses'])

    def step(self, actions):
        obs, reward, done, info = super().step(actions)
        self._collapse_ring = -1
        self._collapse_time = -1

        for ring, collapse in enumerate(self.collapses):

            # NOTE: the following doesn't update the observation directory.
            # This means that the observations will be updated in the next step.
            # update self._collapse_alert_map
            remaining_time = collapse - self._step_count
            if 0 < remaining_time and remaining_time <= 10:
                self._collapse_alert_map = self._alert_collapse(ring, remaining_time)
                self._collapse_ring = ring
                self._collapse_time = remaining_time

            if self._step_count == collapse:
                self._board = self._collapse_board(ring)
                break

        # update the observation accordingly!
        # NOTE: we need to remove bombs, items and agents in the collapsed walls. The observation for these updated state are handled in this self.get_observations()
        obs = self.get_observations()
        return obs, reward, done, info

    def get_observations(self):
        self.observations = self.model.get_observations(
            self._board, self._agents, self._bombs, self._flames,
            self._is_partially_observable, self._agent_view_size,
            self._game_type, self._env)
        for obs in self.observations:
            obs['step_count'] = self._step_count
            obs['collapse_alert_map'] = self._collapse_alert_map  # add collapse alert
            obs['collapse_ring'] = self._collapse_ring  # add collapse alert
            obs['collapse_time'] = self._collapse_time  # add collapse alert
        return self.observations

    def _get_rewards(self):
        '''when ability of an agent increases, it gets reward'''

        # NOTE: we don't use the normal reward anymore
        # reward = super()._get_rewards()
        reward = [0 for _ in self._agents]

        # reward for increase of abilities
        # if an agent get an item that increases its ability, reward 0.1 is given
        if not self._previous_ability:
            self._previous_ability = AbilityTracker(self._agents)
        self._cur_ability = AbilityTracker(self._agents)

        diff = self._cur_ability.diff(self._previous_ability)
        # NOTE: keeping track of ammo is nonsense, because it changes everytime an agents put a bomb.
        for i in range(len(self._agents)):
            if diff[i]['ammo_capacity'] > 0:
                reward[i] += 0.1
            if diff[i]['blast_strength'] > 0:
                reward[i] += 0.1
            if diff[i]['can_kick'] > 0:
                reward[i] += 0.1
        self._previous_ability = self._cur_ability


        # reward for killing/dying
        # if an agent0 is killed:
        #  agent0 (team0) gets reward -2, agent2 (team0) gets reward -1
        #  agent1 (team1) gets reward +1, agent3 (team1) gets reward +1
        assert self._game_type == constants.GameType.Team
        self._cur_is_alive = [agent.is_alive for agent in self._agents]
        who_got_killed = [bool(prev - cur) for prev, cur in zip(self._prev_is_alive, self._cur_is_alive)]
        kill_reward = [0 for _ in self._agents]

        for i, killed in enumerate(who_got_killed):
            # A red team player is killed
            if killed and i % 2 == 0:
                for j, agent in enumerate(self._agents):
                    kill_reward[j] += -1 if j % 2 == 0 else 1
                kill_reward[i] += -1  # penalty

            # A blue team player is killed
            if killed and i % 2 == 1:
                for j, agent in enumerate(self._agents):
                    kill_reward[j] += -1 if j % 2 == 1 else 1
                kill_reward[i] += -1  # penalty

        self._prev_is_alive = self._cur_is_alive


        for i in range(len(reward)):
            reward[i] = reward[i] + kill_reward[i]

        return reward

    # Basically just copied from v0.py
    # But visualize collapse_alert_map in addition

    def _get_done(self):
        done = self.model.get_done(self._agents, self._step_count,
                                   self._max_steps, self._game_type,
                                   self.training_agent)

        # if a training agent dies, the game terminates on the spot.
        alive = [agent for agent in self._agents if agent.is_alive]
        alive_ids = sorted([agent.agent_id for agent in alive])
        if self.training_agent is not None and self.training_agent not in alive_ids:
            done = True
        return done

    def render(self,
               mode=None,
               close=False,
               record_pngs_dir=None,
               record_json_dir=None,
               do_sleep=True):
        if close:
            self.close()
            return

        mode = mode or self._mode or 'human'

        if mode == 'rgb_array':
            rgb_array = graphics.PixelViewer.rgb_array(
                self._board, self._board_size, self._agents,
                self._is_partially_observable, self._agent_view_size)
            return rgb_array[0]

        if self._viewer is None:
            if mode == 'rgb_pixel':
                self._viewer = graphics.PixelViewer(
                    board_size=self._board_size,
                    agents=self._agents,
                    agent_view_size=self._agent_view_size,
                    partially_observable=self._is_partially_observable)
            else:
                self._viewer = graphics.PommeViewer(
                    board_size=self._board_size,
                    agents=self._agents,
                    partially_observable=self._is_partially_observable,
                    agent_view_size=self._agent_view_size,
                    game_type=self._game_type)

            self._viewer.set_board(self._board)
            self._viewer.set_agents(self._agents)
            self._viewer.set_step(self._step_count)
            self._viewer.set_bombs(self._bombs)
            self._viewer.set_flames(self._flames)
            self._viewer.set_collapse_alert_map(self._collapse_alert_map)  # this line is added
            self._viewer.render()

            # Register all agents which need human input with Pyglet.
            # This needs to be done here as the first `imshow` creates the
            # window. Using `push_handlers` allows for easily creating agents
            # that use other Pyglet inputs such as joystick, for example.
            for agent in self._agents:
                if agent.has_user_input():
                    self._viewer.window.push_handlers(agent)
        else:
            self._viewer.set_board(self._board)
            self._viewer.set_agents(self._agents)
            self._viewer.set_step(self._step_count)
            self._viewer.set_bombs(self._bombs)
            self._viewer.set_flames(self._flames)
            self._viewer.set_collapse_alert_map(self._collapse_alert_map)  # this line is added
            self._viewer.render()

        if record_pngs_dir:
            self._viewer.save(record_pngs_dir)
        if record_json_dir:
            self.save_json(record_json_dir)

        if do_sleep:
            time.sleep(1.0 / self._render_fps)
