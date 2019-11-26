'''Entry point into the agents module set'''
from .base_agent import BaseAgent
from .docker_agent import DockerAgent
from .docker_hakozaki_agent import DockerHakozakiAgent
from .http_agent import HttpAgent
from .player_agent import PlayerAgent
from .player_agent_blocking import PlayerAgentBlocking
from .random_agent import RandomAgent
from .simple_agent import SimpleAgent
from .simple_agent_nobomb import SimpleAgentNoBomb
from .static_agent import StaticAgent
from .runaway_agent import RunawayAgent
from .tensorforce_agent import TensorForceAgent
from .multiplayer_over_net_agent import MultiPlayerAgent
