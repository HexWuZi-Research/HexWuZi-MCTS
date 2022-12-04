import numpy as np
import numba as nb
from numba.experimental import jitclass
import random
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
                # if strategy34(self.board, (i, j), -self.player)[0]:
                #     actions = strategy34(self.board, (i, j), -self.player)[1]
                #     break
                # print(strategy34(self.board, (i, j), -self.player))
                if self.board[i, j] == 0 and check_adjacent2(self.board, (i, j)):
                    actions.append((i, j))
        return actions
    
    def take_action(self, action):
        i, j = action
        board = self.board.copy()
        board[i, j] = self.player
        return HexWuZiState(board, -self.player)

    def is_terminal(self):
        winner, gameover = check(self.board)
        return gameover

    def get_reward(self):
        winner, gameover = check(self.board)
        return winner*91/(len(np.nonzero(self.board)[0])-30)



def random_rollout(state: HexWuZiState):
    while not state.is_terminal():
        action = random.choice(state.get_actions())
        state = state.take_action(action)
    return state.get_reward()
