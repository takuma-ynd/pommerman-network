'''This file contains a set of utility functions that
help with positioning, building a game board, and
encoding data to be used later'''
import itertools
import json
import random
import os
from jsonmerge import Merger

from gym import spaces
import numpy as np

from . import constants
from . import agents as pom_agents


class PommermanJSONEncoder(json.JSONEncoder):
    '''A helper class to encode state data into a json object'''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, constants.Item):
            return obj.value
        elif isinstance(obj, constants.Action):
            return obj.value
        elif isinstance(obj, constants.GameType):
            return obj.value
        elif isinstance(obj, np.int64):
            return int(obj)
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif isinstance(obj, spaces.Discrete):
            return obj.n
        elif isinstance(obj, spaces.Tuple):
            return [space.n for space in obj.spaces]
        return json.JSONEncoder.default(self, obj)


def make_board(size, num_rigid=0, num_wood=0, num_agents=4, bomberman_like=False, two_vs_one=False, simple_two_vs_one=False, agent_view_size=3):
    """Make the random but symmetric board.

    The numbers refer to the Item enum in constants. This is:
     0 - passage
     1 - rigid wall
     2 - wood wall
     3 - bomb
     4 - flames
     5 - fog
     6 - extra bomb item
     7 - extra firepower item
     8 - kick
     9 - skull
     10 - 13: agents

    Args:
      size: The dimension of the board, i.e. it's sizeXsize.
      num_rigid: The number of rigid walls on the board. This should be even.
      num_wood: Similar to above but for wood walls.

    Returns:
      board: The resulting random board.
    """

    def lay_wall(value, num_left, coordinates, board):
        '''Lays all of the walls on a board'''
        x, y = random.sample(coordinates, 1)[0]
        coordinates.remove((x, y))
        coordinates.remove((y, x))
        board[x, y] = value
        board[y, x] = value
        num_left -= 2
        return num_left


    def lay_rigid_wall_bomberman_like(coordinates, board):
        '''Lays rigid walls periodically'''
        assert size % 2 == 1
        laid_walls = 0
        for i in range(size):
            for j in range(size):
                outer = (i == 0 or i == size-1 or j == 0 or j == size-1)
                inner = (i % 2 == 0 and j % 2 == 0)
                if outer or inner:
                    board[i,j] = constants.Item.Rigid.value
                    coordinates.discard((i,j))
                    laid_walls += 1
        return laid_walls

    def lay_rigid_wall_two_vs_one(coordinates, board):
        '''Lays rigid walls periodically'''
        assert size % 2 == 1
        laid_walls = 0
        for i in range(size):
            for j in range(size):
                outer = (i == 0 or i == size-1 or j == 0 or j == size-1)
                outer2 = (i == 1 or i == size-2 or j == 1 or j == size-2)
                inner = (i % 2 == 0 and j % 2 == 0)
                if outer or outer2 or inner:
                    board[i,j] = constants.Item.Rigid.value
                    coordinates.discard((i,j))
                    laid_walls += 1
        return laid_walls

    def place_agents_two_vs_one(board):
        '''Procedure:
        1. place enemy agent wherever
        2. decide where to place blocks so that 6 blocks surrounds the enemy
        3. run djikstra to find out cells that are reachable in 4 steps.
        4. place a teammate at one of those cells.
        5. run djikstra again (teammate is not passable) and place another teammate.
        NOTE:
          a. both teammates are in enemy's sight (vise versa)
          b. both teammates can reach the enemy in 4 steps (teammate is not passable!!)
          (this can be implemented by removing coordinates corresponding to the cells in the passage)
        '''
        def in_view_range(position, v_row, v_col, view_size=agent_view_size):
            '''Checks to see if a tile is in an agents viewing area'''
            row, col = position
            return all([
                row >= v_row - view_size, row <= v_row + view_size,
                col >= v_col - view_size, col <= v_col + view_size
            ])
        reachable_distance = 4
        board_size = len(board)

        # prepare coordinates for this function
        _coordinates = set([
            (x, y) for x, y in \
            itertools.product(range(board_size), range(board_size)) \
            if x != y])
        for i in range(board_size):
            for j in range(board_size):
                outer_ridge = (i == 0 or i == board_size-1 or j == 0 or j == board_size-1)
                inner_ridge = (i % 2 == 0 and j % 2 == 0)
                if inner_ridge or outer_ridge:
                    _coordinates.discard((i,j))



        # place the enemy
        enemy_pos_candidates = []
        for i in range(board_size):
            for j in range(board_size):
                manual_pruning = ( i == 2 or i == board_size-1-2 or j == 2 or j == board_size-1-2)
                if (i,j) in _coordinates and not manual_pruning:
                    enemy_pos_candidates.append((i,j))
        enemy_pos = random.choice(enemy_pos_candidates)
        _coordinates.remove(enemy_pos)

        # place blocks at the neighboring free cells
        # NOTE: (i - 1 != 0 and j - 1 != 0) position (corners) need to be always filled
        # otherwise, enemy can be completely trapped
        block_candidates = []
        enemy_surrounding_blocks = []
        block_counts = 0
        for i in range(3):
            for j in range(3):
                posx = enemy_pos[0] - 1 + i
                posy = enemy_pos[1] - 1 + j
                if posx == enemy_pos[0] and posy == enemy_pos[1]: continue
                outer_ridge = (posx == 0 or posx == board_size-1 or posy == 0 or posy == board_size-1)
                inner_ridge = (posx % 2 == 0 and posy % 2 == 0)
                if (outer_ridge or inner_ridge):
                    block_counts += 1
                elif i - 1 != 0 and j -1 != 0:
                    block_counts += 1
                    enemy_surrounding_blocks.append((posx, posy))
                else:
                    block_candidates.append((posx, posy))
        enemy_surrounding_blocks.extend(random.sample(block_candidates, 6 - block_counts))
        _coordinates -= set(enemy_surrounding_blocks)  # subtraction on the set

        # create pre_board
        pre_board = np.zeros((board_size, board_size))
        for i in range(board_size):
            for j in range(board_size):
                outer_ridge = (i == 0 or i == board_size-1 or j == 0 or j == board_size-1)
                inner_ridge = (i % 2 == 0 and j % 2 == 0)
                if outer_ridge or inner_ridge:
                    pre_board[i, j] = constants.Item.Rigid.value
                if (i,j) in enemy_surrounding_blocks:
                    pre_board[i, j] = constants.Item.Wood.value
        # pre_board[enemy_pos] = constants.Item.Agent1.value

        _, dist1, _ = pom_agents.SimpleAgent._djikstra(pre_board, enemy_pos, bombs=[], enemies=[], depth=6)

        # place a teammate: in the sight of the enemy & reachable in 5 steps
        teammate_pos_candidates = []
        for i in range(board_size):
            for j in range(board_size):
                if (i, j) not in _coordinates: continue
                if in_view_range(enemy_pos, i, j) and dist1.get((i,j), np.inf) <= reachable_distance - 2:
                    teammate_pos_candidates.append((i,j))
        teammate_pos = random.choice(teammate_pos_candidates)
        pre_board[teammate_pos] = constants.Item.Wood.value  # NOTE: imaginary wood to prevent teammate2 to 'pass through' teammate1
        _coordinates.remove(teammate_pos)

        # place another teammate: in the sight of enemy and teammate, and reachable to the enemy in 5 steps
        teammate2_pos_candidates = []
        _, dist2, _ = pom_agents.SimpleAgent._djikstra(pre_board, enemy_pos, bombs=[], enemies=[], depth=6)
        for i in range(board_size):
            for j in range(board_size):
                if (i, j) not in _coordinates: continue
                if in_view_range(enemy_pos, i, j) and in_view_range(teammate_pos, i, j, view_size=agent_view_size+1) and reachable_distance - 2 <= dist2.get((i,j), np.inf) and dist2.get((i,j), np.inf) <= reachable_distance:
                    teammate2_pos_candidates.append((i,j))
        teammate2_pos = random.choice(teammate2_pos_candidates)
        _coordinates.remove(teammate2_pos)

        # decide which coordinates to be removed
        # based on cells whose distance is eq to or less than 5 in both 'dist'

        # NOTE: by adding constants.Item.Wood in excludes, we prevent djikstra from calculating path to Wood.
        exclude = [
            constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames, constants.Item.Wood
        ]
        _, dist_enemy_to_t1, _ = pom_agents.SimpleAgent._djikstra(pre_board, enemy_pos, bombs=[], enemies=[], depth=6, exclude=exclude)
        _, dist_t1_to_enemy, _ = pom_agents.SimpleAgent._djikstra(pre_board, teammate_pos, bombs=[], enemies=[], depth=6, exclude=exclude)
        secured_passages = []
        for pos, val in dist_enemy_to_t1.items():
            if val > reachable_distance: continue
            if dist_t1_to_enemy.get(pos, np.inf) > reachable_distance: continue
            secured_passages.append(pos)

        _, dist_enemy_to_t2, _ = pom_agents.SimpleAgent._djikstra(pre_board, enemy_pos, bombs=[], enemies=[], depth=6, exclude=exclude)
        _, dist_t2_to_enemy, _ = pom_agents.SimpleAgent._djikstra(pre_board, teammate2_pos, bombs=[], enemies=[], depth=6, exclude=exclude)
        for pos, val in dist_enemy_to_t2.items():
            if val > reachable_distance: continue
            if dist_t2_to_enemy.get(pos, np.inf) > reachable_distance: continue
            if pos in secured_passages: continue
            secured_passages.append(pos)
        print('secured_passages', secured_passages)
        # agent1 is always the enemy

        # To force fix some weird bug:
        for i in range(3):
            for j in range(3):
                posx = enemy_pos[0] - 1 + i
                posy = enemy_pos[1] - 1 + j
                if (posx, posy) not in secured_passages:
                    secured_passages.append((posx, posy))


        return [teammate_pos, enemy_pos, teammate2_pos], secured_passages, enemy_surrounding_blocks

    def make(size, num_rigid, num_wood, num_agents, two_vs_one=False, simple_two_vs_one=False):
        '''Constructs a game/board'''
        # Initialize everything as a passage.
        board = np.ones((size,
                         size)).astype(np.uint8) * constants.Item.Passage.value

        # Gather all the possible coordinates to use for walls.
        coordinates = set([
            (x, y) for x, y in \
            itertools.product(range(size), range(size)) \
            if x != y])

        # Set the players down. Exclude them from coordinates.
        # Agent0 is in top left. Agent1 is in bottom left.
        # Agent2 is in bottom right. Agent 3 is in top right.
        # assert (num_agents % 2 == 0)

        # if bomberman_like:
        #     agent_positions = [(0,0), (size-1, 0), (size-1, size-1), (0, size-1)]
        # else:
        if two_vs_one:
            agent_positions = [(3, 3), (size - 4, 3), (size - 4, size - 4), (3, size - 4)]
        else:
            agent_positions = [(1, 1), (size - 2, 1), (size - 2, size - 2), (1, size - 2)]

        if simple_two_vs_one:
            positions, secured_passages, enemy_surrounding_woods = place_agents_two_vs_one(board)
            board[positions[0]] = constants.Item.Agent0.value
            board[positions[1]] = constants.Item.Agent1.value
            board[positions[2]] = constants.Item.Agent2.value
            # board[agent_positions[3]] = constants.Item.Agent3.value
            agents = positions

            for position in agents:
                if position in coordinates:
                    coordinates.discard(position)
                    coordinates.discard(position[::-1])  # respect symmetry

            for pos in secured_passages:
                coordinates.discard(pos)
                coordinates.discard(pos[::-1])  # respect symmetry

            for pos in enemy_surrounding_woods:
                board[pos] = constants.Item.Wood.value
                # board[pos[::-1]] = constants.Item.Wood.value  # respect symmetry <-- this possibly overwrite player position
                coordinates.discard(pos)
                coordinates.discard(pos[::-1])  # respect symmetry
                num_wood -= 1

        else:
            if num_agents == 2:
                board[agent_positions[0]] = constants.Item.Agent0.value
                board[agent_positions[-1]] = constants.Item.Agent1.value
                agents = [agent_positions[0], agent_positions[-1]]
            else:
                board[agent_positions[0]] = constants.Item.Agent0.value
                board[agent_positions[1]] = constants.Item.Agent1.value
                board[agent_positions[2]] = constants.Item.Agent2.value
                board[agent_positions[3]] = constants.Item.Agent3.value
                agents = agent_positions

            for position in agents:
                if position in coordinates:
                    coordinates.remove(position)
                    coordinates.remove(position[::-1])  # respect symmetry

        # Exclude breathing room on either side of the agents.
        # if bomberman_like:
            # for i,j in [(0,1), (1,0)]:
            #     coordinates.remove((i,j))
            #     coordinates.remove((size-1 - i, size - 1 - j))
            #     coordinates.remove((size-1 - i, j))
            #     coordinates.remove((i, size-1 - j))
        # else:
        #TEMP:
        if not simple_two_vs_one:
            for i in range(2, 4):
                coordinates.remove((1, i))
                coordinates.remove((i, 1))
                coordinates.remove((size - 2, size - i - 1))
                coordinates.remove((size - i - 1, size - 2))

                if num_agents == 4:
                    coordinates.remove((1, size - i - 1))
                    coordinates.remove((size - i - 1, 1))
                    coordinates.remove((i, size - 2))
                    coordinates.remove((size - 2, i))

        # Lay down wooden walls providing guaranteed passage to other agents.
        wood = constants.Item.Wood.value

        if bomberman_like:
            # for i in range(3, size - 3):
            #     board[0, i] = wood
            #     board[size - i - 1, 0] = wood
            #     board[size - 1, size - i - 1] = wood
            #     board[size - i - 1, size - 1] = wood
            #     coordinates.remove((0, i))
            #     coordinates.remove((size - i - 1, 0))
            #     coordinates.remove((size - 1, size - i - 1))
            #     coordinates.remove((size - i - 1, size - 1))
            #     num_wood -= 4

            # board[2,2] = wood
            # board[2, size-1 - 2] = wood
            # board[size-1 - 2, 2] = wood
            # board[size-1 - 2, size-1 - 2] = wood
            # coordinates.discard((2,2))
            # coordinates.discard((2, size-1 - 2))
            # coordinates.discard((size-1 - 2, 2))
            # coordinates.discard((size-1 - 2, size-1 - 2))
            # num_wood -= 4
            pass
        elif two_vs_one:
            pass
        else:
            if num_agents == 4:
                for i in range(4, size - 4):
                    board[1, i] = wood
                    board[size - i - 1, 1] = wood
                    board[size - 2, size - i - 1] = wood
                    board[size - i - 1, size - 2] = wood
                    coordinates.remove((1, i))
                    coordinates.remove((size - i - 1, 1))
                    coordinates.remove((size - 2, size - i - 1))
                    coordinates.remove((size - i - 1, size - 2))
                    num_wood -= 4

        # Lay down the rigid walls.
        if bomberman_like:
            laid_walls = lay_rigid_wall_bomberman_like(coordinates, board)
            num_rigid -= laid_walls
        elif two_vs_one:
            laid_walls = lay_rigid_wall_two_vs_one(coordinates, board)
            num_rigid -= laid_walls


        while num_rigid > 0:
            num_rigid = lay_wall(constants.Item.Rigid.value, num_rigid,
                                coordinates, board)

        # Lay down the wooden walls.
        while num_wood > 0:
            num_wood = lay_wall(constants.Item.Wood.value, num_wood,
                                coordinates, board)

        return board, agents

    # assert (num_rigid % 2 == 0)
    # assert (num_wood % 2 == 0)
    board, agents = make(size, num_rigid, num_wood, num_agents, two_vs_one=two_vs_one, simple_two_vs_one=simple_two_vs_one)

    # Make sure it's possible to reach most of the passages.
    while len(inaccessible_passages(board, agents)) > 4:
        board, agents = make(size, num_rigid, num_wood, num_agents, two_vs_one=two_vs_one)

    return board


def make_items(board, num_items):
    '''Lays all of the items on the board'''
    item_positions = {}
    while num_items > 0:
        row = random.randint(0, len(board) - 1)
        col = random.randint(0, len(board[0]) - 1)
        if board[row, col] != constants.Item.Wood.value:
            continue
        if (row, col) in item_positions:
            continue

        item_positions[(row, col)] = random.choice([
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]).value
        num_items -= 1
    return item_positions


def inaccessible_passages(board, agent_positions):
    """Return inaccessible passages on this board."""
    seen = set()
    agent_position = agent_positions.pop()
    passage_positions = np.where(board == constants.Item.Passage.value)
    positions = list(zip(passage_positions[0], passage_positions[1]))

    Q = [agent_position]
    while Q:
        row, col = Q.pop()
        for (i, j) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_position = (row + i, col + j)
            if next_position in seen:
                continue
            if not position_on_board(board, next_position):
                continue
            if position_is_rigid(board, next_position):
                continue

            if next_position in positions:
                positions.pop(positions.index(next_position))
                if not len(positions):
                    return []

            seen.add(next_position)
            Q.append(next_position)
    return positions


def is_valid_direction(board, position, direction, invalid_values=None):
    '''Determins if a move is in a valid direction'''
    row, col = position
    if invalid_values is None:
        invalid_values = [item.value for item in \
                          [constants.Item.Rigid, constants.Item.Wood]]

    if constants.Action(direction) == constants.Action.Stop:
        return True

    if constants.Action(direction) == constants.Action.Up:
        return row - 1 >= 0 and board[row - 1][col] not in invalid_values

    if constants.Action(direction) == constants.Action.Down:
        return row + 1 < len(board) and board[row +
                                              1][col] not in invalid_values

    if constants.Action(direction) == constants.Action.Left:
        return col - 1 >= 0 and board[row][col - 1] not in invalid_values

    if constants.Action(direction) == constants.Action.Right:
        return col + 1 < len(board[0]) and \
               board[row][col + 1] not in invalid_values

    raise constants.InvalidAction("We did not receive a valid direction: ",
                                  direction)


def _position_is_item(board, position, item):
    '''Determins if a position holds an item'''
    return board[position] == item.value


def position_is_flames(board, position):
    '''Determins if a position has flames'''
    return _position_is_item(board, position, constants.Item.Flames)


def position_is_bomb(bombs, position):
    """Check if a given position is a bomb.
    
    We don't check the board because that is an unreliable source. An agent
    may be obscuring the bomb on the board.
    """
    for bomb in bombs:
        if position == bomb.position:
            return True
    return False


def position_is_powerup(board, position):
    '''Determins is a position has a powerup present'''
    powerups = [
        constants.Item.ExtraBomb, constants.Item.IncrRange, constants.Item.Kick
    ]
    item_values = [item.value for item in powerups]
    return board[position] in item_values


def position_is_wall(board, position):
    '''Determins if a position is a wall tile'''
    return position_is_rigid(board, position) or \
           position_is_wood(board, position)


def position_is_passage(board, position):
    '''Determins if a position is passage tile'''
    return _position_is_item(board, position, constants.Item.Passage)


def position_is_rigid(board, position):
    '''Determins if a position has a rigid tile'''
    return _position_is_item(board, position, constants.Item.Rigid)


def position_is_wood(board, position):
    '''Determins if a position has a wood tile'''
    return _position_is_item(board, position, constants.Item.Wood)


def position_is_agent(board, position):
    '''Determins if a position has an agent present'''
    return board[position] in [
        constants.Item.Agent0.value, constants.Item.Agent1.value,
        constants.Item.Agent2.value, constants.Item.Agent3.value
    ]


def position_is_enemy(board, position, enemies):
    '''Determins if a position is an enemy'''
    return constants.Item(board[position]) in enemies


# TODO: Fix this so that it includes the teammate.
def position_is_passable(board, position, enemies):
    '''Determins if a possible can be passed'''
    return all([
        any([
            position_is_agent(board, position),
            position_is_powerup(board, position),
            position_is_passage(board, position)
        ]), not position_is_enemy(board, position, enemies)
    ])


def position_is_fog(board, position):
    '''Determins if a position is fog'''
    return _position_is_item(board, position, constants.Item.Fog)


def agent_value(id_):
    '''Gets the state value based off of agents "name"'''
    return getattr(constants.Item, 'Agent%d' % id_).value


def position_in_items(board, position, items):
    '''Dtermines if the current positions has an item'''
    return any([_position_is_item(board, position, item) for item in items])


def position_on_board(board, position):
    '''Determines if a positions is on the board'''
    x, y = position
    return all([len(board) > x, len(board[0]) > y, x >= 0, y >= 0])


def get_direction(position, next_position):
    """Get the direction such that position --> next_position.

    We assume that they are adjacent.
    """
    x, y = position
    next_x, next_y = next_position
    if x == next_x:
        if y < next_y:
            return constants.Action.Right
        else:
            return constants.Action.Left
    elif y == next_y:
        if x < next_x:
            return constants.Action.Down
        else:
            return constants.Action.Up
    raise constants.InvalidAction(
        "We did not receive a valid position transition.")


def get_next_position(position, direction):
    '''Returns the next position coordinates'''
    x, y = position
    if direction == constants.Action.Right:
        return (x, y + 1)
    elif direction == constants.Action.Left:
        return (x, y - 1)
    elif direction == constants.Action.Down:
        return (x + 1, y)
    elif direction == constants.Action.Up:
        return (x - 1, y)
    elif direction == constants.Action.Stop:
        return (x, y)
    raise constants.InvalidAction("We did not receive a valid direction.")


def make_np_float(feature):
    '''Converts an integer feature space into a floats'''
    return np.array(feature).astype(np.float32)


def join_json_state(record_json_dir, agents, finished_at, config, info):
    '''Combines all of the json state files into one'''
    json_schema = {"properties": {"state": {"mergeStrategy": "append"}}}

    json_template = {
        "agents": agents,
        "finished_at": finished_at,
        "config": config,
        "result": {
            "name": info['result'].name,
            "id": info['result'].value
        }
    }

    if info['result'] is not constants.Result.Tie:
        json_template['winners'] = info['winners']

    json_template['state'] = []

    merger = Merger(json_schema)
    base = merger.merge({}, json_template)

    for root, dirs, files in os.walk(record_json_dir):
        for name in files:
            path = os.path.join(record_json_dir, name)
            if name.endswith('.json') and "game_state" not in name:
                with open(path) as data_file:
                    data = json.load(data_file)
                    head = {"state": [data]}
                    base = merger.merge(base, head)

    with open(os.path.join(record_json_dir, 'game_state.json'), 'w') as f:
        f.write(json.dumps(base, sort_keys=True, indent=4))

    for root, dirs, files in os.walk(record_json_dir):
        for name in files:
            if "game_state" not in name:
                os.remove(os.path.join(record_json_dir, name))
