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

# @njit
def strategy34_hor(board, pos, p):
    '''check 34strategy horizontally'''
    strategy_seq5 = [[0,p,p,p,0], [p,p,0,p,p], [0,p,p,p,p], [p,p,p,p,0], [p,0,p,p,p], [p,p,p,0,p]]
    strategy_seq6 = [[0,p,0,p,p,0], [0,p,p,0,p,0]]
    a5 = [[0, 4], [2], [0], [4], [1], [3]]
    a6 = [[0, 2, 5], [0, 3, 5]]
    action = []
    board = np.array(board, dtype=np.intc)
    m5 = len(board) - 5
    m6 = len(board) - 6
    i, j = pos
    if j <= m6:
        if list(board[i, j:j+6]) in strategy_seq6:
            a = strategy_seq6.index(list(board[i, j:j+6]))
            for k in range(len(a6[a])):
                action.append((i, j+a6[a][k]))
            return True, action
        elif list(board[i, j:j+5]) in strategy_seq5:
            a = strategy_seq5.index(list(board[i, j:j+5]))
            for k in range(len(a5[a])):
                action.append((i, j+a5[a][k]))
            return True, action
    
    if j == m5:
        if list(board[i, j:j+5]) in strategy_seq5:
            a = strategy_seq5.index(list(board[i, j:j+5]))
            for k in range(len(a5[a])):
                action.append((i, j+a5[a][k]))
            return True, action
    return False, None


# @njit
def strategy34_ver(board, pos, p):
    '''check 34strategy vertically'''
    strategy_seq5 = [[0,p,p,p,0], [p,p,0,p,p], [0,p,p,p,p], [p,p,p,p,0], [p,0,p,p,p], [p,p,p,0,p]]
    strategy_seq6 = [[0,p,0,p,p,0], [0,p,p,0,p,0]]
    a5 = [[0, 4], [2], [0], [4], [1], [3]]
    a6 = [[0, 2, 5], [0, 3, 5]]
    action = []
    board = np.array(board, dtype=np.intc)
    m5 = len(board) - 5
    m6 = len(board) - 6
    i, j = pos
    if i <= m6:
        if list(board[i:i+6, j]) in strategy_seq6:
            a = strategy_seq6.index(list(board[i:i+6, j]))
            for k in range(len(a6[a])):
                action.append((i+a6[a][k], j))
            return True, action
        elif list(board[i:i+5, j]) in strategy_seq5:
            a = strategy_seq5.index(list(board[i:i+5, j]))
            for k in range(len(a5[a])):
                action.append((i+a5[a][k], j))
            return True, action
    
    if i == m5:
        if list(board[i:i+5, j]) in strategy_seq5:
            a = strategy_seq5.index(list(board[i:i+5, j]))
            for k in range(len(a5[a])):
                action.append((i+a5[a][k], j))
            return True, action
    return False, None

# @njit
def strategy34_inc(board, pos, p):
    '''check inclined 34strateg'''
    strategy_seq5 = [[0,p,p,p,0], [p,p,0,p,p], [0,p,p,p,p], [p,p,p,p,0], [p,0,p,p,p], [p,p,p,0,p]]
    strategy_seq6 = [[0,p,0,p,p,0], [0,p,p,0,p,0]]
    a5 = [[0, 4], [2], [0], [4], [1], [3]]
    a6 = [[0, 2, 5], [0, 3, 5]]
    action = []
    board = np.array(board, dtype=np.intc)
    m5 = len(board) - 5
    m6 = len(board) - 6
    i, j = pos
    if i <= m6 and j <= m6:
        if list(np.diag(board[i:i+6, j:j+6])) in strategy_seq6:
            a = strategy_seq6.index(list(np.diag(board[i:i+6, j:j+6])))
            for k in range(len(a6[a])):
                action.append((i+a6[a][k], j+a6[a][k]))
            return True, action
        elif list(np.diag(board[i:i+5, j:j+5])) in strategy_seq5:
            a = strategy_seq5.index(list(np.diag(board[i:i+5, j:j+5])))
            for k in range(len(a5[a])):
                action.append((i+a5[a][k], j+a5[a][k]))
            return True, action
            
    if i == m5:
        if list(np.diag(board[i:i+5, j:j+5])) in strategy_seq5:
            a = strategy_seq5.index(list(np.diag(board[i:i+5, j:j+5])))
            for k in range(len(a5[a])):
                action.append((i+a5[a][k], j+a5[a][k]))
            return True, action

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
