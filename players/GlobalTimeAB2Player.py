"""
MiniMax Player with AlphaBeta pruning and global time
"""
from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import AlphaBeta
from SearchAlgos import GameState
import SearchAlgos
import time


class Player(AbstractPlayer):
    def __init__(self, game_time):
        AbstractPlayer.__init__(self, game_time) # keep the inheritance of the parent's (AbstractPlayer) __init__()

    def make_move(self, time_limit):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement
        """
        turn_time = self.left_game_time * 0.002
        start = time.time()

        if self.turn < 18:
            move = self._stage_1_move(turn_time - 0.001)
            self.turn += 1

        else:
            move = self._stage_2_move(turn_time - 0.001)
            self.turn += 1

        self.left_game_time -= (time.time() - start)
        return move

    ########## helper functions in class ##########
    def _stage_1_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos)
        alphabeta = AlphaBeta(SearchAlgos.heuristic_stage1, SearchAlgos.succ_stage1, SearchAlgos.goal_func_stage1)
        return self._iterative_deepening(start_state, time_limit, alphabeta)

    def _stage_2_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos)
        alphabeta = AlphaBeta(SearchAlgos.heuristic_stage2, SearchAlgos.succ_stage2, SearchAlgos.is_winning_conf)
        return self._iterative_deepening(start_state, time_limit, alphabeta)


