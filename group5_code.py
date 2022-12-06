from __future__ import division, annotations
import sys
import pygame
import numpy as np
import random
import numba as nb
from numba.experimental import jitclass
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
def check(board, action=None):
    if action is None:
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
    else:
        # ! fast check: just check action piece row/colunm/diag
        i,j = action
        winner = check_single(board[i, :i+6] if i < 6 else board[i, i-5:])
        if winner != 0:
            return winner, True
        winner = check_single(board[:j+6, j] if j < 6 else board[j-5:, j])
        if winner != 0:
            return winner, True
        winner = check_single(np.diag(board, j-i))
        if winner != 0:
            return winner, True
    # check if terminated when nobody win
    if not 0 in board:
        return 0, True
    return 0, False

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

def uct(child: TreeNode, T):
    parent = child.parent
    return parent.state.player * child.reward / child.nvisit + \
        T * np.sqrt(2 * np.log(parent.nvisit) / child.nvisit)

class TreeNode:
    def __init__(self, state: HexWuZiState, parent: TreeNode, action):
        self.state = state
        self.terminal = state.is_terminal(action)[0]
        self.action = action
        self.parent = parent
        self.nvisit = 0
        self.reward = 0
        self.untried_actions = state.get_actions()
        self.children = {}

    def select(self, T):
        node = self
        while not node.terminal:
            # actions is empty, then it's full expanded, then find the best child
            if not node.untried_actions:
                node = node.find_best_child(T)
            else:
                return node
        return node

    def expand(self):
        action = self.untried_actions.pop()
        child = TreeNode(self.state.take_action(action), self, action)
        if child.terminal:
            self.children = {action: child}
            self.untried_actions = None
        else:
            self.children[action] = child
        return child

    def backpropogate(self, reward):
        node = self
        while node:
            node.nvisit += 1
            node.reward += reward
            node = node.parent

    def find_best_child(self, T) -> TreeNode:
        best_value = -np.inf
        best_nodes = []
        for child in self.children.values():
            value = uct(child, T)
            if value > best_value:
                best_value = value
                best_nodes = [child]
            elif value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes)

class MCTS:
    def __init__(self, time_limit=5, T=1/np.sqrt(2),
                 rollout_method=random_rollout):
        self.time_limit = time_limit
        self.T = T
        self.rollout = rollout_method
        self.root: TreeNode = None
        self.our_action = None

    def search(self, state: HexWuZiState, enemy_action=None, need_details=False):
        inherited = False
        # ! try to utilize existed node in the tree
        if enemy_action and self.root:
            node = self.root.children[self.our_action]
            if enemy_action in node.children:
                node = node.children[enemy_action]
                node.parent = None
                inherited = True
        self.root = node if inherited else TreeNode(state, None, None)
        excuted_times = 0
        time_limit = time.time() + self.time_limit
        while time.time() < time_limit:
            self.round()
            excuted_times += 1
        best_child = self.root.find_best_child(0)
        self.our_action = best_child.action
        if need_details:
            return self.our_action, {
                "use_existed_node": inherited,
                "expected_reward": best_child.reward / best_child.nvisit,
                "excuted_times": excuted_times,
                "root": self.root
            }
        else:
            return self.our_action

    def round(self):
        node = self.root.select(self.T)
        if not node.terminal:
            node = node.expand()
        reward = self.rollout(node.state)
        node.backpropogate(reward)

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

def displayboard_people(you_are_black=False, time_limit=5):
    pygame.init()
    # 设置主屏窗口
    screen = pygame.display.set_mode((640,560))
    # 设置窗口的标题，即游戏名称
    pygame.display.set_caption('HexWuZi')

    screen_color= '#E7B941'
    line_color = '#000000'
    screen.fill(screen_color)
    verts = visual_coordinates()
    all_verts = []

    for i in verts:
        for j in i:
            all_verts.append(j)


    display_board = np.zeros((12, 12), dtype=int)
    display_board[0, 1:] = np.arange(0, 11)
    display_board[1:, 0] = np.arange(0, 11)
    broad = empty_broad.copy()
    state = HexWuZiState(broad, 1)
    if you_are_black:
        action = (5,5)
        state = state.take_action(action)
    display_board[1:, 1:] = broad
    print(display_board)
    searcher = MCTS(time_limit=time_limit)

    tim = 0
    flag=False


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


        # input your position

        
        x,y = pygame.mouse.get_pos()
        maybe_position = all_verts[0]
        for i in all_verts:
            if sum((i-np.array([x,y]))**2) < 500:
                pygame.draw.circle(screen, "#808080",i, 18,1)
                maybe_position = i
        pygame.display.update()


        validflag = False
        keys_pressed = pygame.mouse.get_pressed()
        if keys_pressed[0] and tim==0:
            flag = True
            for i in range(len(verts)):
                for j in range(len(verts[0])):
                    if verts[i,j][0] == maybe_position[0] and verts[i,j][1] == maybe_position[1]:
                        maybe_action = (i,j)
            if state.board[maybe_action[0], maybe_action[1]] == 0:
                action = maybe_action
                validflag = True

        if flag:
            tim+=1
        if tim%20==0:#延时200ms
            flag=False
            tim=0

        if validflag:
            print(state.player, action)
            state = state.take_action(action)
            display_board[1:, 1:] = state.board
            print(display_board)
            for i in range(len(state.board)):
                for j in range(len(state.board)):
                    if state.board[i][j] == 1:
                        pygame.draw.circle(screen, "#000000",all_verts[11*i+j], 18,0)
                        pygame.draw.circle(screen, "#808080",all_verts[11*i+j], 18,1)
                    if state.board[i][j] == -1:
                        pygame.draw.circle(screen, "#FFFFFF",all_verts[11*i+j], 18,0)
                        pygame.draw.circle(screen, "#808080",all_verts[11*i+j], 18,1)
            pygame.display.update()
            winner, gameover = check(state.board)
            if gameover:
                if winner == 1:
                    print("You win!")
                    return
                break
            print("AI is searching...")
            action, detail = searcher.search(state=state, enemy_action=action, need_details=True)
            print(state.player, action, detail)
            state = state.take_action(action)
            display_board[1:, 1:] = state.board
            print(display_board)
            winner, gameover = check(state.board)
            if gameover:
                if winner == -1:
                    print("HaHaHaHaHaHaHaHaHaHa! You lose!\n"*5)
                    
                pygame.time.wait(1000)


    return

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

            pygame.time.wait(1000)


def main(you_are_black=False, opponent_name = ""):
    if opponent_name == "group5_code":
        displayBoard()        
    else:
        displayboard_people(you_are_black=you_are_black)
        
if __name__ == "__main__":
    main(you_are_black=False, opponent_name = "group5_code")




