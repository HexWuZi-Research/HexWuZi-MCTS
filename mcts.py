from __future__ import division

import numpy as np
import random
from HexWuZi import *


def uct(child, T):
    parent = child.parent
    return parent.state.player * child.reward / child.nvisit + \
        T * np.sqrt(2 * np.log(parent.nvisit) / child.nvisit)


class TreeNode:
    def __init__(self, state, parent, action):
        self.state = state
        self.terminal = state.is_terminal()
        self.action = action
        self.parent = parent
        self.nvisit = 0
        self.reward = 0
        self.actions = state.get_actions()
        self.children = {}

    def select(self, T):
        node = self
        while not node.terminal:
            # actions is empty, then it's full expanded, then find the best child
            if not node.actions:
                node = node.find_best_child(T)
            else:
                return node
        return node

    def expand(self):
        action = self.actions.pop()
        child = TreeNode(self.state.take_action(action), self, action)
        self.children[action] = child
        return child

    def backpropogate(self, reward):
        node = self
        while node is not None:
            node.nvisit += 1
            node.reward += reward
            node = node.parent

    def find_best_child(self, T):
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

    def search(self, init_state, need_details=False):
        self.root = TreeNode(init_state, None, None)
        excuted_times = 0
        time_limit = time.time() + self.time_limit
        while time.time() < time_limit:
            self.round()
            excuted_times += 1

        best_child = self.root.find_best_child(0)
        action = best_child.action
        if need_details:
            return action, {
                "expectedReward": best_child.reward / best_child.nvisit,
                "excutedTimes": excuted_times,
                "root": self.root
            }
        else:
            return action

    def round(self):
        node = self.root.select(self.T)
        if not node.terminal:
            node = node.expand()
        reward = self.rollout(node.state)
        node.backpropogate(reward)


def self_play():
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
        currentPlayer = state.player
        print(f"AI:{currentPlayer} is searching...")
        action, detail = searcher.search(init_state=state, need_details=True)
        print(action, detail)
        state = state.take_action(action)
        display_board[1:, 1:] = state.board
        print(display_board)
        if state.is_terminal():
            if state.get_reward() == currentPlayer:
                print(f"AI:{currentPlayer} win!")
                return
            break
    print("Draw!")


if __name__ == "__main__":
    self_play()
