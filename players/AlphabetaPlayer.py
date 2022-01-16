"""
MiniMax Player with AlphaBeta pruning
"""
from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import AlphaBeta
from SearchAlgos import GameState
import SearchAlgos

class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None

    ########## helper functions in class ##########
    def _stage_1_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos, self.turn)
        alphabeta = AlphaBeta(SearchAlgos.heuristic_stage1)
        return self._iterative_deepening(start_state, time_limit, alphabeta)

    def _stage_2_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos, self.turn)
        alphabeta = AlphaBeta(SearchAlgos.heuristic_stage2)
        return self._iterative_deepening(start_state, time_limit, alphabeta)
