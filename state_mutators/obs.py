from rlgym.rocket_league import common_values as cv
from rlgym.rocket_league.common_values import BOOST_LOCATIONS
from rlgym.api import ObsBuilder, AgentID

import numpy as np
from typing import List, Dict, Any, Tuple

from rlgym.rocketleague.api import GameState



BALL_SLICE = slice(0, 21)
EGO_SLICE = slice(21, 45)

TEAMMATE_SLICE = [
    slice(45, 81),
    slice(81, 117)
]

OPPONENT_SLICE = [
    slice(117, 153),
    slice(153, 189),
    slice(189, 225)
]

POS = slice(0, 3)
VEL = slice(3, 6)
ANG_VEL = slice(6, 9)
FORWARD = slice(9, 12)
UP = slice(12, 15)
RIGHT = slice(15, 18)
BOOST_AMOUNT = slice(18, 19)
IS_DEMOED = slice(19, 20)
ON_GROUND = slice(20, 21)
HAS_FLIP = slice(21, 22)
IS_SUPERSONIC = slice(22, 23)
DOUBLE_JUMP_TIME = slice(23, 24)

REL_FLIP_POS = slice(24, 27)
REL_FLIP_VEL = slice(27, 30)
REL_CAR_POS  = slice(30, 33)
REL_CAR_VEL  = slice(33, 36)

BALL_POS = slice(0, 3)
BALL_VEL = slice(3, 6)
BALL_ANG_VEL = slice(6, 9)
BALL_REL_FLIP_POS = slice(9, 12)
BALL_REL_FLIP_VEL = slice(12, 15)
BALL_REL_CAR_POS  = slice(15, 18)
BALL_REL_CAR_VEL  = slice(18, 21)



class Obs(ObsBuilder[AgentID, np.array, GameState, Tuple[str, int]]):
    def __init__(self, num_players = 4, num_boosts = 34):
        super().__init__()


        self.last_state




    def get_obs_space(self, agent: AgentID) -> Tuple[str, int]:
        return "real", 100

    def reset(self, agents: AgentID, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_state = initial_state

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]):


        cars = [state.cars[agent_id] for agent_id in agents]
        num_players = len(cars)

        all_cars = np.zeros(num_players, 44)
        all_ball = np.zeros(21)

        for i, c in enumerate(cars):
            phy = c.physics
            all_cars[i, POS] = phy.position
            all_cars[i, VEL] = phy.linear_velocity
            all_cars[i, ANG_VEL] = phy.angular_velocity
            all_cars[i, FORWARD] = phy.forward
            all_cars[i, UP] = phy.up
            all_cars[i, BOOST_AMOUNT] = phy.boost_amount
            all_cars[i, IS_DEMOED] = phy.is_demoed
            all_cars[i, ON_GROUND] = phy.on_ground
            all_cars[i, HAS_FLIP] = phy.has_flip
            all_cars[i, DOUBLE_JUMP_TIME] = 0.0 if (c.air_time_since_jump >= 1.25) else car.air_time_since_jump

        all_cars[:, RIGHT] = np.cross(all_cars[:, FORWARD], all_cars[:, UP])
        spd_sq = np.sum(all_cars[:, VEL] ** 2, axis=1)
        all_cars[:, IS_SUPERSONIC] = (spd_sq >= 4840000.0).astype(np.float32)[:, None]

        player_teams = np.array([c.team_num for c in cars], dtype=int)
        sort_indicies
      





