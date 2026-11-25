from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.common_values import BOOST_LOCATIONS
from rlgym.api import ObsBuilder, AgentID

import numpy as np
from typing import List, Dict, Any, Tuple

from rlgym.rocketleague.api import GameState


class Obs(ObsBuilder[AgentID, np.array, GameState, Tuple[str, int]]):
    def __init__(self, num_players = 4, num_boosts = 34):
        super().__init__()


        self.last_state




    def get_obs_space(self, agent: AgentID) -> Tuple[str, int]:
        return "real", 100

    def reset(self, agents: AgentID, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_state = initial_state

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]):
        pass
      





