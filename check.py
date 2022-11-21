import numpy as np
from numba import njit
import time


@njit
def check_single(line):
    L = len(line)
    for i in range(L-4):
        player = line[i]
        if player != 0 and player == line[i+1] == line[i+2] == line[i+3] == line[i+4]:
            if (i == 0 or line[i-1] != player) and (i == L-5 or line[i+5] != player):
                return player
    return 0

@njit 
def around_check(board,pos):
    END = len(board) - 1
    x,y = pos
    if x != 0 and y != 0 and board[x-1,y-1] in (-1,1):
        return True
    if x != 0 and board[x-1,y]in (-1,1):
        return True
    if y != 0 and board[x,y-1]in (-1,1):
        return True
    if y != END and board[x,y+1] in (-1,1):
        return True
    if x != END and board[x+1,y] in (-1,1):
        return True
    if x!= END and y != END and board[x+1,y+1] in (-1,1):
        return True
    return False



@njit
def check(board):
    # check if anybody win
    for i in range(6):
        winner = check_single(board[i, :i+6])
        if winner != 0:
            return winner, True
        winner = check_single(board[:i+6, i])
        if winner != 0:
            return winner, True
    for i in range(6, 11):
        winner = check_single(board[i, i-5:])
        if winner != 0:
            return winner, True
        winner = check_single(board[i-5:, i])
        if winner != 0:
            return winner, True
    for k in range(-5,6):
        winner = check_single(np.diag(board, k))
        if winner != 0:
            return winner, True
    # check if terminated when nobody win
    if not 0 in board:
        return 0, True
    return 0, False

if __name__ == "__main__":
    board = np.zeros([11, 11]).astype(int)
    for i in range(11//2):
        board[i][11-(5-i):] = 10
        board[10-i][:5-i] = 10

    for i in range(1, 7):
        board[i][i] = -1
    # board[4][1:6] = 1

    print(board)

    print(check(board))

    start_time = time.time()
    for i in range(100000):
        check(board)
    print(time.time()-start_time)
