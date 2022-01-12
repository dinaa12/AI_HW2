"""
MiniMax Player with AlphaBeta pruning with light heuristic
"""
from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import Timeout
import copy
from SearchAlgos import GameState
import SearchAlgos
from SearchAlgos import AlphaBeta
import numpy as np
import time


class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        #TODO: initialize more fields, if needed, and the AlphaBeta algorithm from SearchAlgos.py

    def _iterative_deepening(self, start_state, time_limit, algo):
        ret_val, next_game_state = algo.search(start_state, 4, True, np.inf, time.time())
        cell = next_game_state.player_move
        soldier_that_moved = np.where(self.my_pos != next_game_state.my_pos)[0][0]
        old_cell = self.my_pos[soldier_that_moved]
        self.my_pos[soldier_that_moved] = cell
        self.board[cell] = 1
        if old_cell != -1:
            self.board[old_cell] = 0
        rival_cell = -1 if not self.is_mill(cell) else self._make_mill_get_rival_cell()
        return cell, soldier_that_moved, rival_cell

    def _stage_1_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos)
        alphabeta = AlphaBeta(SearchAlgos.light_heuristic_stage1, SearchAlgos.succ_stage1, SearchAlgos.goal_func_stage1)
        return self._iterative_deepening(start_state, time_limit, alphabeta)

    def _stage_2_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos)
        alphabeta = AlphaBeta(SearchAlgos.light_heuristic_stage2, SearchAlgos.succ_stage2, SearchAlgos.is_winning_conf)
        return self._iterative_deepening(start_state, time_limit, alphabeta)