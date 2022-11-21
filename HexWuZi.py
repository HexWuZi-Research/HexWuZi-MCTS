from __future__ import division

from mcts import mcts
import numpy as np
from check import *


empty_broad = np.zeros([11, 11]).astype(int)
for i in range(11//2):
    empty_broad[i][11-(5-i):] = 10
    empty_broad[10-i][:5-i] = 10


class HexWuZiState():
    def __init__(self, state=empty_broad, player=1):
        self.board = state
        self.currentPlayer = player

    def __str__(self):
        return f"Player:{self.currentPlayer} at board {self.board}"

    def getCurrentPlayer(self):
        return self.currentPlayer

    def getPossibleActions(self):
        possibleActions = []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 0 and around_check(self.board, (i, j)):
                    possibleActions.append(
                        Action(player=self.currentPlayer, x=i, y=j))
        return possibleActions

    def takeAction(self, action):
        newState = HexWuZiState(state=self.board.copy(),
                                player=self.currentPlayer * -1)
        newState.board[action.x][action.y] = action.player
        return newState

    def isTerminal(self):
        winner, gameover = check(self.board)
        return gameover

    def getReward(self):
        winner, gameover = check(self.board)
        return winner


class Action():
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def __str__(self):
        return f"Player:{self.player}, Position:{(self.x, self.y)}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

    def __hash__(self):
        return hash((self.x, self.y, self.player))


def customRollout(state: HexWuZiState):
    I = state.getCurrentPlayer()
    while not state.isTerminal():
        pass


# if __name__ == "__main__":
#     broad = empty_broad.copy()
#     broad[5,5:9]=-1
#     broad[4,5:9]=1
#     print(broad)
#     state = HexWuZiState(state=broad, player=1)
#     searcher = mcts(timeLimit=5)
#     action = searcher.search(initialState=state, needDetails=True)

#     print(action)

def game():
    display_board = np.zeros((12, 12), dtype=int)
    display_board[0, 1:] = np.arange(0, 11)
    display_board[1:, 0] = np.arange(0, 11)
    broad = empty_broad.copy()
    state = HexWuZiState(state=broad, player=1)
    display_board[1:, 1:] = broad
    print(display_board)
    while True:
        text = input("Please input your position: ")
        x, y = text.split(',')
        action = Action(player=1, x=int(x), y=int(y))
        print(action)
        state = state.takeAction(action)
        display_board[1:,1:]=state.board
        print(display_board)
        if state.isTerminal():
            if state.getReward() == 1:
                print("You win!")
            else:
                print("Draw!")
            return
        searcher = mcts(timeLimit=5)
        action = searcher.search(initialState=state)
        print(action)
        state = state.takeAction(action)
        display_board[1:, 1:] = state.board
        print(display_board)
        if state.isTerminal():
            if state.getReward() == -1:
                print("You lose!")
            else:
                print("Draw!")
            return

if __name__ == "__main__":
    game()
