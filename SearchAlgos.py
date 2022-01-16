"""Search Algos: MiniMax, AlphaBeta
"""
import time
import numpy as np
import utils
import copy
ALPHA_VALUE_INIT = -np.inf
BETA_VALUE_INIT = np.inf  # !!!!!


class Timeout(Exception):
    pass


class GameState:
    def __init__(self, board, curr_player, my_pos, rival_pos, turn, player_move=None):
        """
        :param board: board state
        :param curr_player: player who performed last move, 1 - maximizer, 2 - opponent
        :param player_move: position where player placed a solider - new cell
        """
        self.board = board
        self.curr_player = curr_player
        self.player_move = player_move
        self.my_pos = my_pos
        self.rival_pos = rival_pos
        self.turn = turn


### Functions to calc heuristic ###

def is_pos_in_mill(board, pos):
    p = board[pos]
    b = board
    mill = [
        ((b[1] == p and b[2] == p) or (b[3] == p and b[5] == p)),
        ((b[0] == p and b[2] == p) or (b[9] == p and b[17] == p)),
        ((b[0] == p and b[1] == p) or (b[4] == p and b[7] == p)),
        ((b[0] == p and b[5] == p) or (b[11] == p and b[19] == p)),
        ((b[2] == p and b[7] == p) or (b[12] == p and b[20] == p)),
        ((b[0] == p and b[3] == p) or (b[6] == p and b[7] == p)),
        ((b[5] == p and b[7] == p) or (b[14] == p and b[22] == p)),
        ((b[5] == p and b[6] == p) or (b[2] == p and b[4] == p)),
        ((b[9] == p and b[10] == p) or (b[11] == p and b[13] == p)),
        ((b[1] == p and b[17] == p) or (b[8] == p and b[10] == p)),
        ((b[8] == p and b[9] == p) or (b[12] == p and b[15] == p)),
        ((b[3] == p and b[19] == p) or (b[8] == p and b[13] == p)),
        ((b[10] == p and b[15] == p) or (b[4] == p and b[20] == p)),
        ((b[8] == p and b[11] == p) or (b[14] == p and b[15] == p)),
        ((b[13] == p and b[15] == p) or (b[6] == p and b[22] == p)),
        ((b[10] == p and b[12] == p) or (b[13] == p and b[14] == p)),
        ((b[17] == p and b[18] == p) or (b[19] == p and b[21] == p)),
        ((b[1] == p and b[9] == p) or (b[16] == p and b[18] == p)),
        ((b[16] == p and b[17] == p) or (b[20] == p and b[23] == p)),
        ((b[3] == p and b[11] == p) or (b[16] == p and b[21] == p)),
        ((b[4] == p and b[12] == p) or (b[18] == p and b[23] == p)),
        ((b[16] == p and b[19] == p) or (b[22] == p and b[23] == p)),
        ((b[21] == p and b[23] == p) or (b[6] == p and b[14] == p)),
        ((b[18] == p and b[20] == p) or (b[21] == p and b[22] == p))
    ]
    return mill[pos]


def closed_mill(game_state):
    """
    1 if a morris was closed in the last move by the player (and an opponent’s piece should be grabbed in this move),
     -1 if a morris was closed by the opponent in the last move, 0 otherwise
    """
    if is_pos_in_mill(game_state.board, game_state.player_move):
        return 1 if game_state.curr_player == 1 else -1
    else:
        return 0


def is_mill(board, player, pos1, pos2, pos3):
    return board[pos1] == player and board[pos2] == player and board[pos3] == player


def number_of_mills_per_player(game_state, player) -> int:
    """
    :return: number of mills of the player
    """
    res = (is_mill(game_state.board, player, 0, 1, 2)) + \
            (is_mill(game_state.board, player, 8, 9, 10)) + \
            (is_mill(game_state.board, player, 16, 17, 18)) + \
            (is_mill(game_state.board, player, 3, 11, 19)) + \
            (is_mill(game_state.board, player, 20, 12, 4)) + \
            (is_mill(game_state.board, player, 21, 22, 23)) + \
            (is_mill(game_state.board, player, 13, 14, 15)) + \
            (is_mill(game_state.board, player, 5, 6, 7)) + \
            (is_mill(game_state.board, player, 0, 3, 5)) + \
            (is_mill(game_state.board, player, 8, 11, 13)) + \
            (is_mill(game_state.board, player, 16, 19, 21)) + \
            (is_mill(game_state.board, player, 1, 9, 17)) + \
            (is_mill(game_state.board, player, 22, 14, 6)) + \
            (is_mill(game_state.board, player, 18, 20, 23)) + \
            (is_mill(game_state.board, player, 10, 12, 15)) + \
            (is_mill(game_state.board, player, 2, 4, 7))
    return res


def diff_in_number_of_mills(game_state):
    """
    :return: Difference between the number of player's and opponent’s mills
    """
    return int(number_of_mills_per_player(game_state, 1)) - int(number_of_mills_per_player(game_state, 2))


def is_blocked_solider(board, pos):
    directions = utils.get_directions(pos)
    for d in directions:
        if board[d] == 0:
            return False
    return True


def number_of_blocked_soldiers_per_player(game_state, player):
    res = 0
    positions = np.where(game_state.board == player)
    for pos in positions[0]:
        res += is_blocked_solider(game_state.board, pos)
    return res


def diff_in_number_of_blocked_soldiers(game_state):
    """
    :return: Difference between the number of opponent’s and player's blocked soldiers
    """
    return number_of_blocked_soldiers_per_player(game_state, 2) - number_of_blocked_soldiers_per_player(
        game_state, 1)


def diff_in_number_of_soldiers(game_state):
    return len(np.where(game_state.board == 1)[0]) - len(np.where(game_state.board == 2)[0])


def is_incomplete_mill(board, player, pos1, pos2, pos3):
    if ((board[pos1] == 0 and board[pos2] == player and board[pos3] == player) or
            (board[pos2] == 0 and board[pos1] == player and board[pos3] == player) or
            (board[pos3] == 0 and board[pos1] == player and board[pos2] == player)):
        return True
    return False


def number_of_incomplete_mills_per_player(game_state, player):
    return (
            (is_incomplete_mill(game_state.board, player, 0, 1, 2)) +
            (is_incomplete_mill(game_state.board, player, 8, 9, 10)) +
            (is_incomplete_mill(game_state.board, player, 16, 17, 18)) +
            (is_incomplete_mill(game_state.board, player, 3, 11, 19)) +
            (is_incomplete_mill(game_state.board, player, 20, 12, 4)) +
            (is_incomplete_mill(game_state.board, player, 21, 22, 23)) +
            (is_incomplete_mill(game_state.board, player, 13, 14, 15)) +
            (is_incomplete_mill(game_state.board, player, 5, 6, 7)) +
            (is_incomplete_mill(game_state.board, player, 0, 3, 5)) +
            (is_incomplete_mill(game_state.board, player, 8, 11, 13)) +
            (is_incomplete_mill(game_state.board, player, 16, 19, 21)) +
            (is_incomplete_mill(game_state.board, player, 1, 9, 17)) +
            (is_incomplete_mill(game_state.board, player, 22, 14, 6)) +
            (is_incomplete_mill(game_state.board, player, 18, 20, 23)) +
            (is_incomplete_mill(game_state.board, player, 10, 12, 15)) +
            (is_incomplete_mill(game_state.board, player, 2, 4, 7))
    )


def diff_in_number_of_incomplete_mills(game_state):
    """
    :return: Difference between the number of player's and opponent’s 2 piece configurations
    """
    return number_of_incomplete_mills_per_player(game_state, 1) - number_of_incomplete_mills_per_player(game_state, 2)


def is_two_way_incomplete_mill(board, pos):
    p = board[pos]
    if pos == 0:
        return board[1] == p and board[3] == p and board[2] == 0 and board[5] == 0
    if pos == 1:
        return ((board[0] == p and board[9] == p and board[2] == 0 and board[17] == 0) or
                (board[2] == p and board[9] == p and board[0] == 0 and board[17] == 0))
    if pos == 2:
        return board[1] == p and board[4] == p and board[0] == 0 and board[7] == 0
    if pos == 3:
        return ((board[0] == p and board[11] == p and board[5] == 0 and board[19] == 0) or
                (board[5] == p and board[11] == p and board[0] == 0 and board[19] == 0))
    if pos == 4:
        return ((board[7] == p and board[12] == p and board[2] == 0 and board[20] == 0) or
                (board[2] == p and board[12] == p and board[7] == 0 and board[20] == 0))
    if pos == 5:
        return board[3] == p and board[6] == p and board[0] == 0 and board[7] == 0
    if pos == 6:
        return ((board[5] == p and board[14] == p and board[7] == 0 and board[22] == 0) or
                (board[7] == p and board[14] == p and board[5] == 0 and board[22] == 0))
    if pos == 7:
        return board[4] == p and board[6] == p and board[2] == 0 and board[5] == 0
    if pos == 8:
        return board[9] == p and board[11] == p and board[10] == 0 and board[13] == 0
    if pos == 9:
        return ((board[1] == p and board[10] == p and board[17] == 0 and board[8] == 0) or
                (board[10] == p and board[17] == p and board[1] == 0 and board[8] == 0) or
                (board[17] == p and board[8] == p and board[1] == 0 and board[10] == 0) or
                (board[8] == p and board[1] == p and board[10] == 0 and board[17] == 0))
    if pos == 10:
        return board[9] == p and board[12] == p and board[8] == 0 and board[15] == 0
    if pos == 11:
        return ((board[3] == p and board[8] == p and board[13] == 0 and board[19] == 0) or
                (board[8] == p and board[19] == p and board[3] == 0 and board[13] == 0) or
                (board[13] == p and board[19] == p and board[3] == 0 and board[8] == 0) or
                (board[3] == p and board[13] == p and board[8] == 0 and board[19] == 0))
    if pos == 12:
        return ((board[10] == p and board[4] == p and board[15] == 0 and board[20] == 0) or
                (board[4] == p and board[15] == p and board[10] == 0 and board[20] == 0) or
                (board[15] == p and board[20] == p and board[4] == 0 and board[10] == 0) or
                (board[20] == p and board[10] == p and board[4] == 0 and board[15] == 0))
    if pos == 13:
        return board[11] == p and board[14] == p and board[8] == 0 and board[15] == 0
    if pos == 14:
        return ((board[6] == p and board[13] == p and board[15] == 0 and board[22] == 0) or
                (board[13] == p and board[22] == p and board[6] == 0 and board[15] == 0) or
                (board[15] == p and board[22] == p and board[6] == 0 and board[13] == 0) or
                (board[6] == p and board[15] == p and board[13] == 0 and board[22] == 0))
    if pos == 15:
        return board[12] == p and board[14] == p and board[10] == 0 and board[13] == 0
    if pos == 16:
        return board[17] == p and board[19] == p and board[18] == 0 and board[21] == 0
    if pos == 17:
        return ((board[9] == p and board[18] == p and board[1] == 0 and board[16] == 0) or
                (board[9] == p and board[16] == p and board[1] == 0 and board[18] == 0))
    if pos == 18:
        return board[17] == p and board[20] == p and board[16] == 0 and board[23] == 0
    if pos == 19:
        return ((board[11] == p and board[16] == p and board[3] == 0 and board[21] == 0) or
                (board[11] == p and board[21] == p and board[3] == 0 and board[16] == 0))
    if pos == 20:
        return ((board[12] == p and board[18] == p and board[4] == 0 and board[23] == 0) or
                (board[12] == p and board[23] == p and board[4] == 0 and board[18] == 0))
    if pos == 21:
        return board[19] == p and board[22] == p and board[16] == 0 and board[23] == 0
    if pos == 22:
        return ((board[21] == p and board[14] == p and board[6] == 0 and board[23] == 0) or
                (board[14] == p and board[23] == p and board[6] == 0 and board[21] == 0))
    if pos == 23:
        return board[22] == p and board[20] == p and board[18] == 0 and board[21] == 0


def number_of_two_way_incomplete_mill_per_player(game_state, player):
    res = 0
    for pos in range(23):
        if game_state.board[pos] == player:
            res += is_two_way_incomplete_mill(game_state.board, pos)
    return res


def diff_in_number_of_two_way_incomplete_mill(game_state):
    return number_of_two_way_incomplete_mill_per_player(game_state, 1) - \
           number_of_two_way_incomplete_mill_per_player(game_state, 2)


def number_of_double_mills_per_player(game_state, player):
    b = game_state.board
    p = player
    res = 0

    if is_mill(game_state.board, player, 0, 1, 2):
        res += ((b[3] == 0 and b[11] == p and b[19] == p) or
                (b[9] == 0 and b[8] == p and b[10] == p) or
                (b[4] == 0 and b[12] == p and b[20] == p))

    if is_mill(game_state.board, player, 8, 9, 10):
        res += ((b[11] == 0 and b[3] == p and b[19] == p) or
                (b[1] == 0 and b[0] == p and b[2] == p) or
                (b[17] == 0 and b[16] == p and b[18] == p) or
                (b[12] == 0 and b[20] == p and b[4] == p))

    if is_mill(game_state.board, player, 16, 17, 18):
        res += ((b[19] == 0 and b[3] == p and b[11] == p) or
                (b[9] == 0 and b[8] == p and b[10] == p) or
                (b[20] == 0 and b[12] == p and b[4] == p))

    if is_mill(game_state.board, player, 3, 11, 19):
        res += ((b[0] == 0 and b[1] == p and b[2] == p) or
                (b[5] == 0 and b[6] == p and b[7] == p) or
                (b[8] == 0 and b[9] == p and b[10] == p) or
                (b[13] == 0 and b[14] == p and b[15] == p) or
                (b[16] == 0 and b[17] == p and b[18] == p) or
                (b[21] == 0 and b[22] == p and b[23] == p))

    if is_mill(game_state.board, player, 20, 12, 4):
        res += ((b[2] == 0 and b[1] == p and b[0] == p) or
                (b[7] == 0 and b[6] == p and b[5] == p) or
                (b[10] == 0 and b[9] == p and b[8] == p) or
                (b[15] == 0 and b[14] == p and b[13] == p) or
                (b[18] == 0 and b[17] == p and b[16] == p) or
                (b[23] == 0 and b[22] == p and b[21] == p))

    if is_mill(game_state.board, player, 21, 22, 23):
        res += ((b[19] == 0 and b[3] == p and b[11] == p) or
                (b[14] == 0 and b[13] == p and b[15] == p) or
                (b[20] == 0 and b[12] == p and b[4] == p))

    if is_mill(game_state.board, player, 13, 14, 15):
        res += ((b[11] == 0 and b[3] == p and b[19] == p) or
                (b[22] == 0 and b[21] == p and b[23] == p) or
                (b[6] == 0 and b[5] == p and b[7] == p) or
                (b[12] == 0 and b[20] == p and b[4] == p))

    if is_mill(game_state.board, player, 5, 6, 7):
        res += ((b[3] == 0 and b[11] == p and b[19] == p) or
                (b[14] == 0 and b[13] == p and b[15] == p) or
                (b[4] == 0 and b[12] == p and b[20] == p))

    if is_mill(game_state.board, player, 0, 3, 5):
        res += ((b[1] == 0 and b[9] == p and b[17] == p) or
                (b[11] == 0 and b[8] == p and b[13] == p) or
                (b[6] == 0 and b[14] == p and b[22] == p))

    if is_mill(game_state.board, player, 8, 11, 13):
        res += ((b[9] == 0 and b[1] == p and b[17] == p) or
                (b[3] == 0 and b[0] == p and b[5] == p) or
                (b[19] == 0 and b[16] == p and b[21] == p) or
                (b[14] == 0 and b[6] == p and b[22] == p))

    if is_mill(game_state.board, player, 16, 19, 21):
        res += ((b[17] == 0 and b[9] == p and b[1] == p) or
                (b[11] == 0 and b[8] == p and b[13] == p) or
                (b[22] == 0 and b[14] == p and b[6] == p))

    if is_mill(game_state.board, player, 1, 9, 17):
        res += ((b[0] == 0 and b[3] == p and b[5] == p) or
                (b[2] == 0 and b[4] == p and b[7] == p) or
                (b[8] == 0 and b[11] == p and b[13] == p) or
                (b[10] == 0 and b[12] == p and b[15] == p) or
                (b[16] == 0 and b[19] == p and b[21] == p) or
                (b[18] == 0 and b[20] == p and b[23] == p))

    if is_mill(game_state.board, player, 22, 14, 6):
        res += ((b[5] == 0 and b[3] == p and b[0] == p) or
                (b[7] == 0 and b[4] == p and b[2] == p) or
                (b[13] == 0 and b[11] == p and b[8] == p) or
                (b[15] == 0 and b[12] == p and b[10] == p) or
                (b[21] == 0 and b[19] == p and b[16] == p) or
                (b[23] == 0 and b[20] == p and b[18] == p))

    if is_mill(game_state.board, player, 18, 20, 23):
        res += ((b[17] == 0 and b[9] == p and b[1] == p) or
                (b[12] == 0 and b[10] == p and b[15] == p) or
                (b[22] == 0 and b[14] == p and b[6] == p))

    if is_mill(game_state.board, player, 10, 12, 15):
        res += ((b[9] == 0 and b[1] == p and b[17] == p) or
                (b[20] == 0 and b[18] == p and b[23] == p) or
                (b[4] == 0 and b[2] == p and b[7] == p) or
                (b[14] == 0 and b[6] == p and b[22] == p))

    if is_mill(game_state.board, player, 2, 4, 7):
        res += ((b[1] == 0 and b[9] == p and b[17] == p) or
                (b[12] == 0 and b[10] == p and b[15] == p) or
                (b[6] == 0 and b[14] == p and b[22] == p))

    return res


def diff_in_number_of_double_mills(game_state):
    return number_of_double_mills_per_player(game_state, 1) - number_of_double_mills_per_player(game_state, 2)


def is_winning_conf(game_state):
    """
    :return: 1 if the state is winning for the player, -1 if losing, 0 otherwise
    """
    if game_state.turn < 18:
        return 0
    if game_state.curr_player == 1:
        if (len(np.where(game_state.board == 2)[0]) < 3 or
                number_of_blocked_soldiers_per_player(game_state, 2) == len(np.where(game_state.board == 2)[0])):
            return 1
        else:
            return 0
    else:
        if (len(np.where(game_state.board == 1)[0]) < 3 or
                number_of_blocked_soldiers_per_player(game_state, 1) == len(np.where(game_state.board == 1)[0])):
            return -1
        else:
            return 0


def heuristic_stage1(game_state):
    h = 18 * closed_mill(game_state) + 26 * diff_in_number_of_mills(game_state) + \
        1 * diff_in_number_of_blocked_soldiers(game_state) + 6 * diff_in_number_of_soldiers(game_state) + \
        12 * diff_in_number_of_incomplete_mills(game_state) + 7 * diff_in_number_of_two_way_incomplete_mill(game_state)
    return h


def heuristic_stage2(game_state):
    h = 14 * closed_mill(game_state) + 43 * diff_in_number_of_mills(game_state) + \
        10 * diff_in_number_of_blocked_soldiers(game_state) + 8 * diff_in_number_of_soldiers(game_state) + \
        42 * diff_in_number_of_double_mills(game_state) + 1086 * is_winning_conf(game_state)
    return h


def light_heuristic_stage1(game_state):
    h = 25 * diff_in_number_of_mills(game_state) + 20 * diff_in_number_of_mills(game_state)
    return h


def light_heuristic_stage2(game_state):
    h = 25 * diff_in_number_of_mills(game_state) + 1000 * is_winning_conf(game_state)
    return h


### succ funcs ###
def choose_succ_func(game_state):
        if game_state.turn < 18:
            return succ_stage1(game_state)
        else:
            return succ_stage2(game_state)


def succ_stage1(game_state):
    new_game_state = copy.deepcopy(game_state)

    for cell in range(23):
        if new_game_state.board[cell] == 0:
            new_game_state.board[cell] = new_game_state.curr_player
            new_game_state.player_move = cell

            if new_game_state.curr_player == 1:
                num_of_soldier = np.where(new_game_state.my_pos == -1)[0][0]
                new_game_state.my_pos[num_of_soldier] = cell
            else:
                num_of_soldier = np.where(new_game_state.rival_pos == -1)[0][0]
                new_game_state.rival_pos[num_of_soldier] = cell

            new_game_state.curr_player = 3 - new_game_state.curr_player
            new_game_state.turn += 1

            yield new_game_state

            new_game_state.turn -= 1
            new_game_state.curr_player = 3 - new_game_state.curr_player

            if new_game_state.curr_player == 1:
                new_game_state.my_pos[num_of_soldier] = -1
            else:
                new_game_state.rival_pos[num_of_soldier] = -1

            new_game_state.board[cell] = 0


def succ_stage2(game_state):
    new_game_state = copy.deepcopy(game_state)

    for cell in range(23):
        if new_game_state.board[cell] == new_game_state.curr_player:
            new_game_state.board[cell] = 0
            for d in utils.get_directions(cell):
                if new_game_state.board[d] == 0:
                    new_game_state.board[d] = new_game_state.curr_player
                    new_game_state.player_move = d

                    if new_game_state.curr_player == 1:
                        num_of_soldier = np.where(new_game_state.my_pos == cell)[0][0]
                        new_game_state.my_pos[num_of_soldier] = d
                    else:
                        num_of_soldier = np.where(new_game_state.rival_pos == cell)[0][0]
                        new_game_state.rival_pos[num_of_soldier] = d

                    new_game_state.curr_player = 3 - new_game_state.curr_player
                    new_game_state.turn += 1

                    yield new_game_state

                    new_game_state.turn -= 1
                    new_game_state.curr_player = 3 - new_game_state.curr_player

                    new_game_state.board[d] = 0
                    if new_game_state.curr_player == 1:
                        new_game_state.my_pos[num_of_soldier] = cell
                    else:
                        new_game_state.rival_pos[num_of_soldier] = cell

            new_game_state.board[cell] = new_game_state.curr_player


### goal func ###
def goal_func_stage1(game_state):
    return len(np.where(game_state.my_pos == -1)[0]) == 0 and len(np.where(game_state.rival_pos == -1)[0]) == 0


class SearchAlgos:
    def __init__(self, utility):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The successor function.
        """
        self.utility = utility
        self.succ = choose_succ_func
        self.goal = is_winning_conf

    def search(self, game_state, depth, maximizing_player, time_limit, start_time):
        pass


class MiniMax(SearchAlgos):

    def search(self, game_state, depth, maximizing_player, time_limit, start_time):
        """Start the MiniMax algorithm.
        :param start_time:
        :param time_limit:
        :param game_state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        if time.time() - start_time >= time_limit:
            raise Timeout

        if abs(self.goal(game_state)) or depth == 0:
            if maximizing_player:
                return self.utility(game_state), game_state
            else:
                return self.utility(game_state), None

        children = self.succ(game_state)

        if maximizing_player:
            cur_max = -np.inf
            next_game_state = None
            for c in children:
                ret_val = self.search(c, depth-1, not maximizing_player, time_limit, start_time)
                if ret_val[0] > cur_max:
                    cur_max = ret_val[0]
                    next_game_state = copy.deepcopy(c)
            return cur_max, next_game_state

        else:
            cur_min = np.inf
            for c in children:
                value = self.search(c, depth-1, not maximizing_player, time_limit, start_time)[0]
                cur_min = min(value, cur_min)
            return cur_min, None


class AlphaBeta(SearchAlgos):

    def search(self, game_state, depth, maximizing_player, time_limit, start_time, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param turn:
        :param time_limit:
        :param start_time:
        :param game_state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        if time.time() - start_time >= time_limit:
            raise Timeout

        if abs(self.goal(game_state)) or depth == 0:
            if maximizing_player:
                return self.utility(game_state), game_state
            else:
                return self.utility(game_state), None

        children = self.succ(game_state)

        if maximizing_player:
            cur_max = -np.inf
            next_game_state = None
            for c in children:
                ret_val = self.search(c, depth - 1, not maximizing_player, time_limit, start_time, alpha, beta)
                if ret_val[0] > cur_max:
                    cur_max = ret_val[0]
                    next_game_state = copy.deepcopy(c)
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return np.inf, next_game_state
            return cur_max, next_game_state

        else:
            cur_min = np.inf
            for c in children:
                value = self.search(c, depth-1, not maximizing_player, time_limit, start_time, alpha, beta)[0]
                cur_min = min(value, cur_min)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return -np.inf, None
            return cur_min, None
