from __future__ import division
import sys
import pygame
import numpy as np
from HexWuZi import *
from mcts import *

def visual_coordinates():
    L = 5 # number of layers
    XY = np.full((1+2*L, 1+2*L, 2), np.nan)

    R = (3**.5) / 2
    dxNewLayer = [None, -.5, .5, 1, .5, -.5, -1]
    dyNewLayer = [None, R, R, 0, -R, -R, 0]
    dxSameLayer = [None, 1, .5, -.5, -1, -.5, .5]
    dySameLayer = [None, 0, -R, -R, 0, R, R]
    diNewLayer = [None, 1, 1, 0, -1, -1, 0]
    djNewLayer = [None, 0, 1, 1, 0, -1, -1]
    diSameLayer = [None, 0, -1, -1, 0, 1, 1]
    djSameLayer = [None, 1, 0, -1, -1, 0, 1]
    
    i, j = L, L # stone positions
    x, y = 0, 0 # visual coordinates
    XY[i, j] = x, y
    for n in range(L):        # build a new layer
        for u in range(1, 7): # define a corner
            i = (n+1)*diNewLayer[u] + 5
            j = (n+1)*djNewLayer[u] + 5
            x = (n+1)*dxNewLayer[u]
            y = (n+1)*dyNewLayer[u]
            XY[i, j] = [x, y]
            for v in range(1, n+1): # go from the corner clockwise
                i += diSameLayer[u] # to the next corner
                j += djSameLayer[u]
                x += dxSameLayer[u]
                y += dySameLayer[u]
                XY[i, j] = [x, y]
    XY *= 60
    XY[..., 0] += 320 # x
    XY[..., 1] += 280 # y
    return XY

def displayBoard():
    pygame.init()
    # 设置主屏窗口
    screen = pygame.display.set_mode((640,560))
    # 设置窗口的标题，即游戏名称
    pygame.display.set_caption('HexWuZi')
    verts = visual_coordinates()
    all_verts = []

    for i in verts:
        for j in i:
            all_verts.append(j)

    screen_color= '#E7B941'
    line_color = '#000000'
    screen.fill(screen_color)
    display_board = np.zeros((12, 12), dtype=int)
    display_board[0, 1:] = np.arange(0, 11)
    display_board[1:, 0] = np.arange(0, 11)
    broad = empty_broad.copy()
    broad[5, 5] = 1
    state = HexWuZiState(broad, -1)
    display_board[1:, 1:] = broad
    print(display_board)
    searcher = MCTS(time_limit=5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.fill(screen_color)
        for i in all_verts:
            for j in all_verts:
                if sum((i-j)**2) < 3900:
                    pygame.draw.line(screen,line_color,i,j,2)

        for i in range(len(state.board)):
            for j in range(len(state.board)):
                if state.board[i][j] == 1:
                    pygame.draw.circle(screen, "#000000",all_verts[11*i+j], 18,0)
                    pygame.draw.circle(screen, "#808080",all_verts[11*i+j], 18,1)
                if state.board[i][j] == -1:
                    pygame.draw.circle(screen, "#FFFFFF",all_verts[11*i+j], 18,0)
                    pygame.draw.circle(screen, "#808080",all_verts[11*i+j], 18,1)
        pygame.display.flip()

        currentPlayer = state.player
        print(f"AI:{currentPlayer} is searching...")
        action, detail = searcher.search(state, need_details=True)
        print(action, detail)
        state = state.take_action(action)
        display_board[1:, 1:] = state.board
        print(display_board)
        winner, gameover = check(state.board)
        if gameover:
            if winner == currentPlayer:
                print(f"AI:{currentPlayer} win!")
                return
            break




if __name__ == "__main__":
    displayBoard()
