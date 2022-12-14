import numpy as np
import numba as nb
from numba.experimental import jitclass
# import random
from check import *

empty_broad = np.zeros([11, 11], dtype=np.intc)
for i in range(11//2):
    empty_broad[i][11-(5-i):] = 10
    empty_broad[10-i][:5-i] = 10


@jitclass([('board', nb.intc[:, :]), ('player', nb.intc)])
class HexWuZiState:
    def __init__(self, board, player):
        self.board = board
        self.player = player

    def get_actions(self):
        actions = []
        for i in range(6):
            for j in range(i+6):
                if self.board[i, j] == 0 and check_adjacent(self.board, (i, j)):
                    actions.append((i, j))
        for i in range(6, 11):
            for j in range(i-5, 11):
                if self.board[i, j] == 0 and check_adjacent(self.board, (i, j)):
                    actions.append((i, j))
        return actions

    def take_action(self, action):
        i, j = action
        board = self.board.copy()
        board[i, j] = self.player
        return HexWuZiState(board, -self.player)

    def is_terminal(self, action=None):
        winner, gameover = check(self.board, action)
        return gameover, winner


@njit
def depth_reward(winner, board):
    return winner*91/(np.count_nonzero(board)-30)


@njit
def random_rollout(state: HexWuZiState):
    action = None
    while True:
        gameover, winner = state.is_terminal(action)
        if gameover:
            return depth_reward(winner, state.board)
        actions = state.get_actions()
        action = actions[np.random.randint(len(actions))]
        # action = random.choice(actions)
        state = state.take_action(action)
