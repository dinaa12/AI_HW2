"""Search Algos: MiniMax, AlphaBeta
"""
#TODO: you can import more modules, if needed
#TODO: update ALPHA_VALUE_INIT, BETA_VALUE_INIT in utils
import time
import numpy as np
ALPHA_VALUE_INIT = -np.inf
BETA_VALUE_INIT = np.inf # !!!!!

class SearchAlgos:
    def __init__(self, succ, perform_move=None, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = self.heuristic
        self.succ = succ
        self.perform_move = perform_move
        self.goal = goal

    def search(self, state, depth, maximizing_player):
        pass

    def heuristic(self, state):
        h = .....
        return h


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        if self.goal(state) or depth == 0:
            if maximizing_player:
                return self.utility(state), 000   # TODO: replace 000, "direction in case of max node"???
            else:
                return self.utility(state), None

        children = self.succ(state, 2 - maximizing_player)

        if maximizing_player:
            cur_max = -np.inf
            for c in children:
                value = self.search(c, depth-1, not maximizing_player)
                cur_max = max(value, cur_max)
            return cur_max

        else:
            cur_min = np.inf
            for c in children:
                value = self.search(c, depth-1, not maximizing_player)
                cur_min = min(value, cur_min)
            return cur_min



class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        if self.goal(state) or depth == 0:
            if maximizing_player:
                return self.utility(state), 000   # TODO: replace 000, "direction in case of max node"???
            else:
                return self.utility(state), None

        children = self.succ(state, 2 - maximizing_player)

        if maximizing_player:
            cur_max = -np.inf
            for c in children:
                value = self.search(c, depth - 1, not maximizing_player, alpha, beta)
                cur_max = max(value, cur_max)
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return np.inf
            return cur_max

        else:
            cur_min = np.inf
            for c in children:
                value = self.search(c, depth-1, not maximizing_player, alpha, beta)
                cur_min = min(value, cur_min)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return -np.inf
            return cur_min


