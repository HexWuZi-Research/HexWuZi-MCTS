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
def check_adjacent(board, pos):
    """Check whether given position is adjacent to exist pieces"""
    END = len(board) - 1
    x, y = pos
    if x != 0 and board[x-1, y] in (-1, 1):
        return True
    if y != 0 and board[x, y-1] in (-1, 1):
        return True
    if x != 0 and y != 0 and board[x-1, y-1] in (-1, 1):
        return True
    if y != END and board[x, y+1] in (-1, 1):
        return True
    if x != END and board[x+1, y] in (-1, 1):
        return True
    if x != END and y != END and board[x+1, y+1] in (-1, 1):
        return True
    return False

@njit
def check_adjacent2(board, pos):
    ''' almost the same as check_adjacent()
        but extend the range to 2 laps'''
    x, y = pos
    END = len(board) - 2
    if check_adjacent(board, pos):
        return True
    else:
        if x > 1 and board[x-2, y] in (-1, 1):
            return True
        if y > 1 and board[x, y-2] in (-1, 1):
            return True
        if x > 0 and y > 0 and board[x-2, y-2] in (-1, 1):
            return True
        if y < END and board[x, y+2] in (-1, 1):
            return True
        if x < END and board[x+2, y] in (-1, 1):
            return True
        if x < END and y < END and board[x+2, y+2] in (-1, 1):
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
    for k in range(-5, 6):
        winner = check_single(np.diag(board, k))
        if winner != 0:
            return winner, True
    # check if terminated when nobody win
    if not 0 in board:
        return 0, True
    return 0, False

@njit
def check_strategy(line, p):
    if list(line).count(p) >= 4 and list(line).count(-p) == 0:
        position = np.where(line == 0)[0]
        return True, position
    
    if list(line).count(p) == 3 and list(line).count(-p) == 0:
        position = np.where(line == 0)[0]
        m = len(line) - 1
        if 0 in list(position) and m in list(position) and line[m-1] == p and line[1] == p:
            return True, position
        else:
            return False, None
        
    return False, None

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
    for k in range(-5, 6):
        winner = check_single(np.diag(board, k))
        if winner != 0:
            return winner, True
    # check if terminated when nobody win
    if not 0 in board:
        return 0, True
    return 0, False


@njit
def strategy34(board, pos, p):
    i, j = pos
    actions = []
    if j < len(board) - 6:
        hor5 = board[i, j:j+5]
        hor6 = board[i, j:j+6]
        if check_strategy(hor5, p)[0]:
            position = check_strategy(hor5, p)[1]
            for k in range(len(position)):
                actions.append((i, j+position[k]))
            return True, actions
        if check_strategy(hor6, p)[0]:
            position = check_strategy(hor6, p)[1]
            for k in range(len(position)):
                actions.append((i, j+position[k]))
            return True, actions
    
    if i < len(board) - 6:
        ver5 = board[i:i+5, j]
        ver6 = board[i:i+6, j]
        if check_strategy(ver5, p)[0]:
            position = check_strategy(ver5, p)[1]
            for k in range(len(position)):
                actions.append((i+position[k], j))
            return True, actions
        if check_strategy(ver6, p)[0]:
            position = check_strategy(ver6, p)[1]
            for k in range(len(position)):
                actions.append((i+position[k], j))
            return True, actions
    
    if i < len(board) - 6 and j < len(board) - 6:
        inc5 = np.diag(board[i:i+5, j:j+5])
        inc6 = np.diag(board[i:i+6, j:j+6])
        if check_strategy(inc5, p)[0]:
            position = check_strategy(inc5, p)[1]
            for k in range(len(position)):
                actions.append((i+position[k], j+position[k]))
            return True, actions
        if check_strategy(inc6, p)[0]:
            position = check_strategy(inc6, p)[1]
            for k in range(len(position)):
                actions.append((i+position[k], j+position[k]))
            return True, actions
    return False, None


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
