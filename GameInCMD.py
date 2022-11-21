from __future__ import division

from mcts import mcts
import numpy as np
from HexWuZi import *


def game():
    display_board = np.zeros((12, 12), dtype=int)
    display_board[0, 1:] = np.arange(0, 11)
    display_board[1:, 0] = np.arange(0, 11)
    broad = empty_broad.copy()
    state = HexWuZiState(state=broad, player=1)
    display_board[1:, 1:] = broad
    print(display_board)
    while True:
        while True:
            text = input("Please input your position: ")
            if "," in text:
                x, y = text.split(',')
            elif " " in text:
                x, y = text.split(" ")
            else:
                print('Please input position in form: "x,y" or "x y"!')
                continue
            try:
                x,y = int(x),int(y)
            except ValueError:
                print("Please input postion in integer!")
                continue
            if state.board[x,y] == 0:
                action = Action(player=1, x=x, y=y)
                break
            print(f"({x},{y}) is not available! Input again")
        print(action)
        state = state.takeAction(action)
        display_board[1:,1:]=state.board
        print(display_board)
        if state.isTerminal():
            if state.getReward() == 1:
                print("You win!")
                return
            break
        print("AI is searching...")
        searcher = mcts(timeLimit=5)
        action,detail = searcher.search(initialState=state,needDetails=True)
        print(action,detail)
        state = state.takeAction(action)
        display_board[1:, 1:] = state.board
        print(display_board)
        if state.isTerminal():
            if state.getReward() == -1:
                print("You lose!")
                return
            break
    print("Draw!")

if __name__ == "__main__":
    game()