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
        #TODO: initialize more fields, if needed, and the AlphaBeta algorithm from SearchAlgos.py
        self.board = None
        self.my_pos = None
        self.rival_pos = None
        self.turn = 0


    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, of the board.
        No output is expected.
        """
        self.board = board
        self.my_pos = np.full(9, -1)
        self.rival_pos = np.full(9, -1)
        self.turn = 0

    def make_move(self, time_limit):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement
        """

        if self.turn < 18:
            move = self._stage_1_move(time_limit)
            self.turn += 1
            return move

        else:
            move = self._stage_2_move(time_limit)
            self.turn += 1
            return move

    def set_rival_move(self, move):
        """Update your info, given the new position of the rival.
        input:
            - move: tuple, the new position of the rival.
        No output is expected
        """
        rival_pos, rival_soldier, my_dead_pos = move

        if self.turn < 18:
            self.board[rival_pos] = 2
            self.rival_pos[rival_soldier] = rival_pos
        else:
            rival_prev_pos = self.rival_pos[rival_soldier]
            self.board[rival_prev_pos] = 0
            self.board[rival_pos] = 2
            self.rival_pos[rival_soldier] = rival_pos
        if my_dead_pos != -1:
            self.board[my_dead_pos] = 0
            dead_soldier = int(np.where(self.my_pos == my_dead_pos)[0][0])
            self.my_pos[dead_soldier] = -2
        self.turn += 1

    ########## helper functions in class ##########
    def _choose_rival_cell_to_kill(self):
        rival_cell = np.where(self.board == 2)[0][0]
        return rival_cell

    def _make_mill_get_rival_cell(self):
        rival_cell = self._choose_rival_cell_to_kill()
        rival_idx = np.where(self.rival_pos == rival_cell)[0][0]
        self.rival_pos[rival_idx] = -2
        self.board[rival_cell] = 0
        return rival_cell

    def _iterative_minimax(self, start_state, time_limit, minimax):
        start = time.time()
        depth = 1
        ret_val, cell, new_my_pos = minimax.search(start_state, depth, True, time_limit, start)
       # cell, new_my_pos = direction
        soldier_that_moved = np.where(self.my_pos != new_my_pos)[0]

        try:
            while time.time() - start < time_limit:
                ret_val, cell, new_my_pos = minimax.search(start_state, depth, True, time_limit, start)
                #cell, new_my_pos = direction
                depth += 1
        except Timeout:
            soldier_that_moved = np.where(self.my_pos != new_my_pos)[0]
            self.my_pos[soldier_that_moved] = cell
            self.board[cell] = 1

        rival_cell = -1 if not self.is_mill(cell) else self._make_mill_get_rival_cell()
        return cell, soldier_that_moved, rival_cell

    def _stage_1_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 2, self.my_pos, self.rival_pos)
        minimax = MiniMax(SearchAlgos.heuristic_stage1, SearchAlgos.succ_stage1, SearchAlgos.goal_func_stage1)
        return self._iterative_minimax(start_state, time_limit, minimax)

    def _stage_2_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos)
        minimax = MiniMax(SearchAlgos.heuristic_stage2, SearchAlgos.succ_stage2, SearchAlgos.is_winning_conf)
        return self._iterative_minimax(start_state, time_limit, minimax)

