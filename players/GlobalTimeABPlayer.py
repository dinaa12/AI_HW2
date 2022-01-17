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
        #turn_time = self.left_game_time * 0.2
        start = time.time()

        if self.turn < 18:
            turn_time = self.left_game_time * 0.1
            move = self._stage_1_move(turn_time - 0.01)
            self.turn += 1

        else:
            turn_time = self.left_game_time * 0.2
            move = self._stage_2_move(turn_time - 0.01)
            self.turn += 1

        self.left_game_time -= (time.time() - start)
        return move

    ########## helper functions in class ##########
    def _stage_1_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos, self.turn)
        alphabeta = AlphaBeta()
        return self._iterative_deepening(start_state, time_limit, alphabeta)

    def _stage_2_move(self, time_limit) -> tuple:
        start_state = GameState(self.board, 1, self.my_pos, self.rival_pos, self.turn)
        alphabeta = AlphaBeta()
        return self._iterative_deepening(start_state, time_limit, alphabeta)


