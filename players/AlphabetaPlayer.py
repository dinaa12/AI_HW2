"""
MiniMax Player with AlphaBeta pruning
"""
from players.AbstractPlayer import AbstractPlayer
#TODO: you can import more modules, if needed
import time
import numpy as np
from SearchAlgos import AlphaBeta
from SearchAlgos import Timeout
from SearchAlgos import GameState
import SearchAlgos

class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        #TODO: initialize more fields, if needed, and the AlphaBeta algorithm from SearchAlgos.py
        self.board = None
        self.my_pos = None
        self.rival_pos = None
        self.turn = 0

    ########## helper functions in class ##########
    def _stage_1_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos)
        alphabeta = AlphaBeta(SearchAlgos.heuristic_stage1, SearchAlgos.succ_stage1, SearchAlgos.goal_func_stage1)
        return self._iterative_deepening(start_state, time_limit, alphabeta)

    def _stage_2_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos)
        alphabeta = AlphaBeta(SearchAlgos.heuristic_stage2, SearchAlgos.succ_stage2, SearchAlgos.is_winning_conf)
        return self._iterative_deepening(start_state, time_limit, alphabeta)
