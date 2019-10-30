"""Module to handle all of the graphics components.

'rendering' converts a display specification (such as :0) into an actual
Display object. Pyglet only supports multiple Displays on Linux.
"""
from datetime import datetime
import math
import os
from random import randint
from time import strftime

from gym.utils import reraise
import numpy as np
from PIL import Image

try:
    import pyglet
except ImportError as error:
    reraise(
        suffix="Install pyglet with 'pip install pyglet'. If you want to just "
        "install all Gym dependencies, run 'pip install -e .[all]' or "
        "'pip install gym[all]'.")

try:
    from pyglet.gl import *
    LAYER_BACKGROUND = pyglet.graphics.OrderedGroup(0)
    LAYER_FOREGROUND = pyglet.graphics.OrderedGroup(1)
    LAYER_TOP = pyglet.graphics.OrderedGroup(2)
    LAYER_TOPPER = pyglet.graphics.OrderedGroup(3)
    LAYER_OVERLAY = pyglet.graphics.OrderedGroup(4)
except pyglet.canvas.xlib.NoSuchDisplayException as error:
    print("Import error NSDE! You will not be able to render --> %s" % error)
except ImportError as error:
    print("Import error GL! You will not be able to render --> %s" % error)

from . import constants
from . import utility

__location__ = os.path.dirname(os.path.realpath(__file__))
RESOURCE_PATH = os.path.join(__location__, constants.RESOURCE_DIR)


class Viewer(object):
    ''' Base class for the graphics module.
        Used to share common functionality between the different
        rendering engines.
     '''
    def __init__(self):
        self.window = None
        self.display = None
        self._agents = []
        self._agent_count = 0
        self._board_state = None
        self._batch = None
        self._selected_action = None
        self._step = 0
        self._waiting = False
        self._yourturn = False
        self._collapse_alert_map = None
        self._gameover = False
        self._message = ''
        self._agent_view_size = None
        self._is_partially_observable = False
        self.isopen = False

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def set_board(self, state):
        self._board_state = state

    def set_bombs(self, bombs):
        self._bombs = bombs

    def set_flames(self, flames):
        self._flames= flames

    def set_agents(self, agents):
        self._agents = agents
        self._agent_count = len(agents)

    def set_step(self, step):
        self._step = step

    def set_selected_action(self, action):
        self._selected_action = action

    def set_waiting(self, waiting):
        self._waiting = waiting

    def set_collapse_alert_map(self, alert_map):
        self._collapse_alert_map = alert_map

    def set_gameover(self):
        self._gameover = True

    def set_message(self, message):
        self._message = message

    def close(self):
        self.window.close()
        self.isopen = False

    def window_closed_by_user(self):
        self.isopen = False

    def save(self, path):
        now = datetime.now()
        filename = now.strftime('%m-%d-%y_%H-%M-%S_') + str(
            self._step) + '.png'
        path = os.path.join(path, filename)
        pyglet.image.get_buffer_manager().get_color_buffer().save(path)


class PixelViewer(Viewer):
    '''Renders the game as a set of square pixels'''
    def __init__(self,
                 display=None,
                 board_size=11,
                 agents=[],
                 partially_observable=False,
                 agent_view_size=None,
                 game_type=None):
        super().__init__()
        from gym.envs.classic_control import rendering
        self.display = rendering.get_display(display)
        self._board_size = board_size
        self._agent_count = len(agents)
        self._agents = agents
        self._is_partially_observable = partially_observable
        self._agent_view_size = agent_view_size

    def render(self):
        frames = self.build_frame()

        if self.window is None:
            height, width, _channels = frames.shape
            self.window = pyglet.window.Window(
                width=4 * width,
                height=4 * height,
                display=self.display,
                vsync=False,
                resizable=True)
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                '''Registers an event handler with a pyglet window to resize the window'''
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                ''' Registers an event handler with a pyglet to tell the render engine the
                    window is closed
                '''
                self.isopen = True

        assert len(frames.shape
                  ) == 3, "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            frames.shape[1],
            frames.shape[0],
            'RGB',
            frames.tobytes(),
            pitch=frames.shape[1] * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0, width=self.window.width, height=self.window.height)
        self.window.flip()

    def build_frame(self):
        board = self._board_state
        board_size = self._board_size
        agents = self._agents
        human_factor = constants.HUMAN_FACTOR
        rgb_array = self.rgb_array(board, board_size, agents,
                                   self._is_partially_observable,
                                   self._agent_view_size)

        all_img = np.array(Image.fromarray(rgb_array[0].astype(np.uint8)).resize(
            (board_size * human_factor, board_size * human_factor), resample=Image.NEAREST))
        other_imgs = [
            np.array(Image.fromarray(frame.astype(np.uint8)).resize(
                (int(board_size * human_factor / len(self._agents)),
                 int(board_size * human_factor / len(self._agents))),
                resample=Image.NEAREST)) for frame in rgb_array[1:]
        ]

        other_imgs = np.concatenate(other_imgs, 0)
        img = np.concatenate([all_img, other_imgs], 1)

        return img

    @staticmethod
    def rgb_array(board, board_size, agents, is_partially_observable,
                  agent_view_size):
        frames = []

        all_frame = np.zeros((board_size, board_size, 3))
        num_items = len(constants.Item)
        for row in range(board_size):
            for col in range(board_size):
                value = board[row][col]
                if utility.position_is_agent(board, (row, col)):
                    num_agent = value - num_items + 4
                    if agents[num_agent].is_alive:
                        all_frame[row][col] = constants.AGENT_COLORS[num_agent]
                else:
                    all_frame[row][col] = constants.ITEM_COLORS[value]

        all_frame = np.array(all_frame)
        frames.append(all_frame)

        for agent in agents:
            row, col = agent.position
            my_frame = all_frame.copy()
            for r in range(board_size):
                for c in range(board_size):
                    if is_partially_observable and not all([
                            row >= r - agent_view_size, row <
                            r + agent_view_size, col >= c - agent_view_size,
                            col < c + agent_view_size
                    ]):
                        my_frame[r, c] = constants.ITEM_COLORS[
                            constants.Item.Fog.value]
            frames.append(my_frame)

        return frames


class PommeViewer(Viewer):
    '''The primary render engine for pommerman.'''
    def __init__(self,
                 display=None,
                 board_size=11,
                 agents=[],
                 partially_observable=False,
                 agent_view_size=None,
                 game_type=None,
                 ):
        super().__init__()
        from gym.envs.classic_control import rendering
        self.display = rendering.get_display(display)
        board_height = constants.TILE_SIZE * board_size
        height = math.ceil(board_height + (constants.BORDER_SIZE * 2) +
                           (constants.MARGIN_SIZE * 3))
        ability_width = constants.TILE_SIZE * 2
        width = math.ceil(board_height + ability_width +
                          (constants.BORDER_SIZE * 2) + constants.MARGIN_SIZE)
        # board_height + (constants.BORDER_SIZE * 2)

        self._height = height
        self._width = width

        self.window = pyglet.window.Window(
                width=width, height=height, display=display)

        self.window.set_caption('Pommerman')

        self.isopen = True
        self._board_size = board_size
        self._resource_manager = ResourceManager(game_type)
        self._tile_size = constants.TILE_SIZE
        self._agent_tile_size = (board_height / 4) / board_size
        self._agent_abilities_size = constants.TILE_SIZE * 2
        self._agent_count = len(agents)
        self._agents = agents
        self._game_type = game_type
        self._is_partially_observable = partially_observable
        self._agent_view_size = agent_view_size

        @self.window.event
        def close(self):
            '''Pyglet event handler to close the window'''
            self.window.close()
            self.isopen = False


    def render(self):
        self.window.switch_to()
        self.window.dispatch_events()
        self._batch = pyglet.graphics.Batch()

        # TEMP:
        agent_id = 0

        background = self.render_background()
        text = self.render_text()
        agents = self.render_dead_alive()
        board = self.render_main_board(agent_id=agent_id)
        abilities = self.render_abilities(agent_id=agent_id, size=self._tile_size)
        if self._selected_action is not None:
            action = self.render_selected_action(agent_id=agent_id, size=self._tile_size)

        # TODO: fix this!
        # NOTE: This is very tricky!
        # by overwriting the variable: waiting, render_waiting will be overwritten by render_message!
        if self._waiting:
            waiting = self.render_waiting()

        # if self._yourturn:
        #     waiting = self.render_yourturn()

        # if self._gameover:
        #     waiting = self.render_gameover()

        if self._message != '':
            waiting = self.render_message(self._message)
        # agents_board = self.render_agents_board()

        self._batch.draw()
        self.window.flip()

    def render_main_board(self, agent_id=None):
        board = self._board_state
        size = self._tile_size
        x_offset = constants.BORDER_SIZE
        y_offset = constants.BORDER_SIZE
        top = self.board_top(-constants.BORDER_SIZE - 8)
        return self.render_board(board, x_offset, y_offset, size, top)

    def render_agents_board(self):
        x_offset = self._board_size * self._tile_size + constants.BORDER_SIZE
        x_offset += constants.MARGIN_SIZE
        size = self._agent_tile_size
        agents = []
        top = self._height - constants.BORDER_SIZE + constants.MARGIN_SIZE
        for agent in self._agents:
            y_offset = agent.agent_id * size * self._board_size + (
                agent.agent_id * constants.MARGIN_SIZE) + constants.BORDER_SIZE
            agent_board = self.agent_view(agent)
            sprite = self.render_board(agent_board, x_offset, y_offset, size,
                                       top)
            agents.append(sprite)
        return agents

    def render_abilities(self, agent_id, size):
        def get_abilities():
            agent = self._agents[agent_id]
            return {'ammo': agent.ammo,
                    'strength': agent.blast_strength,
                    'kick': agent.can_kick}

        def draw_ability(fig_id, x, y, val):
            # create the tile
            tile = self._resource_manager.tile_from_state_value(fig_id)
            tile.width = size; tile.height = size
            sprite = pyglet.sprite.Sprite(
                        tile, x, y, batch=self._batch, group=LAYER_FOREGROUND)

            text = pyglet.text.Label(
                str(val) if type(val) is not bool else '',
                font_name='Cousine-Regular',
                font_size=30,
                x=x + self._tile_size + constants.MARGIN_SIZE,
                y=y,
                batch=self._batch,
                group=LAYER_TOP)
            text.color = constants.TILE_COLOR

            return sprite, text


        if agent_id is None:
            agent_id = 0

        x_offset = self._board_size * self._tile_size + constants.BORDER_SIZE
        x_offset += constants.MARGIN_SIZE
        top = self.board_top(-constants.BORDER_SIZE - 8)
        y_offset = top - constants.BORDER_SIZE

        abilities = get_abilities()
        sprite_ammo = draw_ability(6, x_offset, y_offset, abilities['ammo'])
        sprite_bomb_strength = draw_ability(7, x_offset, y_offset - size, abilities['strength'] - 1)

        sprites = [sprite_ammo, sprite_bomb_strength]
        if abilities['kick']:
            sprite_kick = draw_ability(8, x_offset, y_offset - size * 2, abilities['kick'])
            sprites.append(sprite_kick)

        return sprites

    def render_board(self, board, x_offset, y_offset, size, top=0):
        def draw_bomb_life(x, y, blast_strength, color=constants.BOMB_LIFE_COLOR, opacity=255, group=LAYER_TOP):
            strength = pyglet.text.Label(
                str(blast_strength),
                font_name='Arial',
                font_size=20,
                x=x + 17,
                y=y + 12,
                batch=self._batch,
                group=group)
            strength.color = color
            strength.opacity = opacity
            return strength

        sprites = []
        blast_strength = []
        flame_life = []
        mov_dirs = []
        bomb_positions = [bomb.position for bomb in self._bombs]
        overlay_bombs = []
        collapse_alerts = []

        for row in range(self._board_size):
            for col in range(self._board_size):
                x = col * size + x_offset
                y = top - y_offset - row * size
                tile_state = board[row][col]
                if tile_state == constants.Item.Bomb.value:

                    # draw a bomb according to its blast strength
                    bomb = self.get_bomb(row, col)
                    tile = self._resource_manager.get_bomb_blast_tile(bomb.blast_strength)

                    # draw bomb's life
                    life = draw_bomb_life(x, y, bomb.life)
                    blast_strength.append(life)

                    # draw bomb's moving direction if it's been kicked
                    if bomb.moving_direction is not None:
                        # 1: up
                        # 2: down
                        # 3: left
                        # 4: right
                        strings = ['↑', '↓', '←', '→']
                        arrow = strings[bomb.moving_direction.value - 1]
                        direction = pyglet.text.Label(
                            arrow,
                            font_name='Arial',
                            font_size=32,
                            bold=True,
                            x=x + 14,
                            y=y + 12,
                            batch=self._batch,
                            group=LAYER_TOP)
                        direction.color = constants.MOVING_DIR_COLOR
                        mov_dirs.append(direction)


                elif tile_state == constants.Item.Flames.value:
                    # draw a flame
                    tile = self._resource_manager.tile_from_state_value(tile_state)

                    # draw the remaining time on the flame
                    flame = self.get_flame(row, col)
                    life = pyglet.text.Label(
                        str(flame.life + 1),
                        font_name='Arial',
                        font_size=20,
                        x=x + 17,
                        y=y + 12,
                        batch=self._batch,
                        group=LAYER_TOP)
                    life.color = constants.FLAME_LIFE_COLOR
                    flame_life.append(life)

                elif tile_state in constants.Item.Agents.value:
                    tile = self._resource_manager.tile_from_state_value(tile_state)
                    # import ipdb; ipdb.set_trace()
                    if (row, col) in bomb_positions:
                        # draw the life on the bomb
                        bomb = self.get_bomb(row, col)
                        strength = draw_bomb_life(x, y, bomb.life, color=constants.BOMB_LIFE_COLOR, opacity=constants.OVERLAY_OPACITY)

                        overlay = self._resource_manager.get_bomb_blast_tile(bomb.blast_strength)
                        overlay.width = size
                        overlay.height = size
                        overlay_sprite = pyglet.sprite.Sprite(
                            overlay, x, y, batch=self._batch, group=LAYER_TOP)
                        overlay_sprite.opacity = constants.OVERLAY_OPACITY
                        overlay_bombs.append(overlay_sprite)

                else:
                    tile = self._resource_manager.tile_from_state_value(tile_state)

                tile.width = size
                tile.height = size
                sprite = pyglet.sprite.Sprite(
                    tile, x, y, batch=self._batch, group=LAYER_FOREGROUND)
                sprites.append(sprite)

                # draw collapse alert if the tile is not rigid and not fog.
                if tile_state != constants.Item.Rigid.value and tile_state != constants.Item.Fog.value:
                    if self._collapse_alert_map is not None:
                        alert = self._collapse_alert_map[row, col]
                        if alert > 0:
                            life = draw_bomb_life(x, y, int(alert), color=(255, 100, 100, 200), group=LAYER_TOPPER)
                            overlay = self._resource_manager.tile_from_state_value(constants.Item.Rigid.value)
                            overlay.width = size
                            overlay.height = size
                            overlay_sprite = pyglet.sprite.Sprite(
                                overlay, x, y, batch=self._batch, group=LAYER_TOPPER)
                            overlay_sprite.opacity = constants.OVERLAY_OPACITY - 60
                            collapse_alerts.append(overlay_sprite)


        return sprites, blast_strength, overlay_bombs, collapse_alerts

    def render_selected_action(self, agent_id, size):
        assert self._selected_action is not None

        def draw_action(act, x, y):
            # create the tile
            tile = self._resource_manager.get_action_tile(act)
            tile.width = size; tile.height = size
            sprite = pyglet.sprite.Sprite(
                        tile, x, y, batch=self._batch, group=LAYER_FOREGROUND)

            return sprite

        x_offset = self._board_size * self._tile_size + constants.BORDER_SIZE
        x_offset += constants.MARGIN_SIZE
        top = self.board_top(-constants.BORDER_SIZE - 8)
        y_offset = constants.BORDER_SIZE
        sprite_action = draw_action(self._selected_action, x_offset, y_offset)

        return sprite_action


    def agent_view(self, agent):
        if not self._is_partially_observable:
            return self._board_state

        agent_view_size = self._agent_view_size
        state = self._board_state.copy()
        fog_value = self._resource_manager.fog_value()
        row, col = agent.position

        for r in range(self._board_size):
            for c in range(self._board_size):
                if self._is_partially_observable and not all([
                        row >= r - agent_view_size, row <= r + agent_view_size,
                        col >= c - agent_view_size, col <= c + agent_view_size
                ]):
                    state[r][c] = fog_value

        return state

    def render_background(self):
        image_pattern = pyglet.image.SolidColorImagePattern(
            color=constants.BACKGROUND_COLOR)
        image = image_pattern.create_image(self._width, self._height)
        return pyglet.sprite.Sprite(
            image, 0, 0, batch=self._batch, group=LAYER_BACKGROUND)

    def render_waiting(self):
        assert self._waiting
        # hacky way to put gray shade on the game screen
        fog_image = self._resource_manager.fog_tile()
        waiting_shade = pyglet.sprite.Sprite(
            fog_image,
            0,
            0,
            batch=self._batch,
            group=LAYER_OVERLAY)
        waiting_shade.scale = 20
        waiting_shade.color = (0,0,0)  # black
        waiting_shade.opacity = 150

        message = 'Waiting for other players...'
        waiting_text = pyglet.text.Label(
            message,
            font_name='Cousine-Regular',
            font_size=14,
            x=constants.BORDER_SIZE,
            # y=self._board_size * self._tile_size // 2,
            y=self._tile_size,
            batch=self._batch,
            group=LAYER_TOP)
        waiting_text.color = constants.TILE_COLOR
        waiting_text.opacity = 200
        return waiting_shade, waiting_text

    def render_message(self, message):
        text = pyglet.text.Label(
            message,
            font_name='Cousine-Regular',
            font_size=30,
            x=constants.BORDER_SIZE + self._tile_size * 2,
            # y=self._board_size * self._tile_size // 2,
            y=constants.BORDER_SIZE + self._tile_size * 6,
            batch=self._batch,
            group=LAYER_TOP)
        text.color = constants.TILE_COLOR
        text.opacity = 200
        return text

    def render_gameover(self):
        message = 'Game Over'
        waiting_text = pyglet.text.Label(
            message,
            font_name='Cousine-Regular',
            font_size=30,
            x=constants.BORDER_SIZE + self._tile_size * 4,
            # y=self._board_size * self._tile_size // 2,
            y=constants.BORDER_SIZE + self._tile_size * 6,
            batch=self._batch,
            group=LAYER_TOP)
        waiting_text.color = constants.TILE_COLOR
        waiting_text.opacity = 200
        return waiting_text


    def render_text(self):
        text = []
        board_top = self.board_top(y_offset=8)
        title_label = pyglet.text.Label(
            'Pommerman',
            font_name='Cousine-Regular',
            font_size=36,
            x=constants.BORDER_SIZE,
            y=board_top,
            batch=self._batch,
            group=LAYER_TOP)
        title_label.color = constants.TILE_COLOR
        text.append(title_label)

        info_text = ''
        if self._game_type is not None:
            info_text += 'Mode: ' + self._game_type.name + '   '

        info_text += 'Time: ' + strftime('%b %d, %Y %H:%M:%S')
        info_text += '   Step: ' + str(self._step)

        time_label = pyglet.text.Label(
            info_text,
            font_name='Arial',
            font_size=10,
            x=constants.BORDER_SIZE,
            y=5,
            batch=self._batch,
            group=LAYER_TOP)
        time_label.color = constants.TEXT_COLOR
        text.append(time_label)
        return text

    def render_dead_alive(self):
        board_top = self.board_top(y_offset=5)
        image_size = 30
        spacing = 5
        dead = self._resource_manager.dead_marker()
        dead.width = image_size
        dead.height = image_size
        sprites = []
        
        if self._game_type is constants.GameType.FFA or self._game_type is constants.GameType.OneVsOne:
            agents = self._agents
        else:
            agents = [self._agents[i] for i in [0,2,1,3]]

        for index, agent in enumerate(agents):
            # weird math to make sure the alignment
            # is correct. 'image_size + spacing' is an offset
            # that includes padding (spacing) for each image. 
            # '4 - index' is used to space each agent out based
            # on where they are in the array based off of their
            # index. 
            x = self.board_right() - (len(agents) - index) * (
                image_size + spacing)
            y = board_top
            agent_image = self._resource_manager.agent_image(agent.agent_id)
            agent_image.width = image_size
            agent_image.height = image_size
            sprites.append(
                pyglet.sprite.Sprite(
                    agent_image,
                    x,
                    y,
                    batch=self._batch,
                    group=LAYER_FOREGROUND))

            if agent.is_alive is False:
                sprites.append(
                    pyglet.sprite.Sprite(
                        dead, x, y, batch=self._batch, group=LAYER_TOP))

        return sprites

    def board_top(self, y_offset=0):
        return constants.BORDER_SIZE + (
            self._board_size * self._tile_size) + y_offset

    def board_right(self, x_offset=0):
        return constants.BORDER_SIZE + (
            self._board_size * self._tile_size) + x_offset

    def get_bomb_properties(self, row, col):
        for bomb in self._bombs:
            x, y = bomb.position
            if x == row and y == col:
                return bomb.life, bomb.blast_strength, bomb.moving_direction

    def get_bomb(self, row, col):
        for bomb in self._bombs:
            x, y = bomb.position
            if x == row and y == col:
                return bomb

    def get_flame(self, row, col):
        # multiple flames can exist in a cell
        # return the one that has the longest life
        coexisting_flames = []
        for flame in self._flames:
            x, y = flame.position
            if x == row and y == col:
                coexisting_flames.append(flame)

        longest_life = -1
        for flame in coexisting_flames:
            if flame.life > longest_life:
                longest_life_flame = flame
                longest_life = longest_life_flame.life
        return longest_life_flame





class ResourceManager(object):
    '''Handles sprites and other resources for the PommeViewer'''
    def __init__(self, game_type):
        self._index_resources()
        self._load_fonts()
        self.images = self._load_images()
        self.bombs = self._load_bombs()
        self.blast_bombs = self._load_blast_bombs()
        self.actions = self._load_actions()
        self._fog_value = self._get_fog_index_value()
        self._is_team = True

        if game_type == constants.GameType.FFA or game_type == constants.GameType.OneVsOne:
            self._is_team = False

    @staticmethod
    def _index_resources():
        # Tell pyglet where to find the resources
        pyglet.resource.path = [RESOURCE_PATH]
        pyglet.resource.reindex()

    @staticmethod
    def _load_images():
        images_dict = constants.IMAGES_DICT
        for i in range(0, len(images_dict)):
            image_data = images_dict[i]
            image = pyglet.resource.image(image_data['file_name'])
            images_dict[i]['image'] = image

        return images_dict

    @staticmethod
    def _load_bombs():
        images_dict = constants.BOMB_DICT
        for i in range(0, len(images_dict)):
            image_data = images_dict[i]
            image = pyglet.resource.image(image_data['file_name'])
            images_dict[i]['image'] = image

        return images_dict

    @staticmethod
    def _load_actions():
        images_dict = constants.ACTION_DICT
        for i in range(0, len(images_dict)):
            image_data = images_dict[i]
            image = pyglet.resource.image(image_data['file_name'])
            images_dict[i]['image'] = image

        return images_dict

    @staticmethod
    def _load_blast_bombs():
        images_dict = constants.BLAST_BOMB_DICT
        for i in range(0, len(images_dict)):
            image_data = images_dict[i]
            image = pyglet.resource.image(image_data['file_name'])
            images_dict[i]['image'] = image

        return images_dict

    @staticmethod
    def _load_fonts():
        for i in range(0, len(constants.FONTS_FILE_NAMES)):
            font_path = os.path.join(RESOURCE_PATH,
                                     constants.FONTS_FILE_NAMES[i])
            pyglet.font.add_file(font_path)

    @staticmethod
    def _get_fog_index_value():
        for id, data in constants.IMAGES_DICT.items():
            if data['name'] == 'Fog':
                return id

    def tile_from_state_value(self, value):
        if self._is_team and value in range(10, 14):
            return self.images[value + 10]['image']

        return self.images[value]['image']

    def agent_image(self, agent_id):
        if self._is_team:
            return self.images[agent_id + 24]['image']

        return self.images[agent_id + 15]['image']

    def dead_marker(self):
        return self.images[19]['image']

    def fog_value(self):
        return self._fog_value

    def fog_tile(self):
        img = self.images[self._fog_value]
        return img['image']

    def get_bomb_tile(self, life):
        return self.bombs[life - 1]['image']

    def get_bomb_blast_tile(self, blast_strength):
        if blast_strength > 10:
            return self.blast_bombs[-1]['image']
        return self.blast_bombs[blast_strength - 2]['image']

    def get_action_tile(self, action):
        return self.actions[action]['image']
