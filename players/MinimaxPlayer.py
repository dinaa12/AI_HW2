"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
#TODO: you can import more modules, if needed
import time
import numpy as np
from SearchAlgos import MiniMax
from SearchAlgos import Timeout
from SearchAlgos import GameState
import SearchAlgos


class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None

    ########## helper functions in class ##########
    def _stage_1_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos, self.turn)
        minimax = MiniMax()
        return self._iterative_deepening(start_state, time_limit, minimax)

    def _stage_2_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos, self.turn)
        minimax = MiniMax()
        return self._iterative_deepening(start_state, time_limit, minimax)

