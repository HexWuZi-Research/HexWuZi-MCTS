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
        M, N = self.board.shape
        actions = []
        for i in range(M):
            for j in range(N):
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
        return gameover

    def get_reward(self, action=None):
        winner, gameover = check(self.board, action)
        return winner*91/(len(np.nonzero(self.board)[0])-30)


@njit
def random_rollout(state: HexWuZiState):
    action = None
    while not state.is_terminal(action):
        actions = state.get_actions()
        action = actions[np.random.randint(len(actions))]
        # action = random.choice(actions)
        state = state.take_action(action)
    return state.get_reward(action)
