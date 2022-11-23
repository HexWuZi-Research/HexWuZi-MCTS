import numpy as np
from argparse import ArgumentParser
from mcts import *


def game(time_limit=5):
    display_board = np.zeros((12, 12), dtype=int)
    display_board[0, 1:] = np.arange(0, 11)
    display_board[1:, 0] = np.arange(0, 11)
    broad = empty_broad.copy()
    state = HexWuZiState(broad, 1)
    display_board[1:, 1:] = broad
    print(display_board)
    searcher = MCTS(time_limit=time_limit)
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
                x, y = int(x), int(y)
            except ValueError:
                print("Please input postion in integer!")
                continue
            if state.board[x, y] == 0:
                action = (x, y)
                break
            print(f"({x},{y}) is not available! Input again")
        print(state.player, action)
        state = state.take_action(action)
        display_board[1:, 1:] = state.board
        print(display_board)
        winner, gameover = check(state.board)
        if gameover:
            if winner == 1:
                print("You win!")
                return
            break
        print("AI is searching...")
        action, detail = searcher.search(init_state=state, need_details=True)
        print(state.player, action, detail)
        state = state.take_action(action)
        display_board[1:, 1:] = state.board
        print(display_board)
        winner, gameover = check(state.board)
        if gameover:
            if winner == -1:
                print("You lose!")
                return
            break
    print("Draw!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--time_limit", type=float, default=5.,
                        help="set time limit for AI")
    args = parser.parse_args()
    game(time_limit=args.time_limit)
