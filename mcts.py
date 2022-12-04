from __future__ import division, annotations

import numpy as np
import random
from HexWuZi import *


def uct(child: TreeNode, T):
    parent = child.parent
    return parent.state.player * child.reward / child.nvisit + \
        T * np.sqrt(2 * np.log(parent.nvisit) / child.nvisit)


class TreeNode:
    def __init__(self, state: HexWuZiState, parent: TreeNode, action):
        self.state = state
        self.terminal = state.is_terminal(action)
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
