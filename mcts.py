import math
import random

import numpy as np

import config
import game.othello as othello
import nn_util


class Edge:
    def __init__(self, move, parent_node):
        self.parent_node = parent_node
        self.move = move
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = 0


class Node:
    def __init__(self, board: othello.Board, parent_edge: Edge):
        self.board = board
        self.parent_edge = parent_edge
        self.child_edge_node = []

    def expand(self, network):
        moves = self.board.legal_moves()
        for m in moves:
            child_board = self.board.__copy__()
            child_board.apply_move(m)
            child_edge = Edge(m, self)
            child_node = Node(child_board, child_edge)
            self.child_edge_node.append((child_edge, child_node))

        q = network.predict(np.array([nn_util.board_to_nn_input(self.board)]))
        prob_sum = 0.

        for (edge, node) in self.child_edge_node:
            edge.P = q[0][0][edge.move.pos]
            prob_sum += edge.P
        for edge, _ in self.child_edge_node:
            edge.P /= prob_sum
        v = q[1][0][0]
        return v

    def is_leaf_node(self):
        return self.child_edge_node == []


class MCTS:
    def __init__(self, network):
        self.network = network
        self.root_node = None
        self.tau = 1.0
        self.c_puct = 1.0

    def utc_value(self, edge, parent_N):
        return self.c_puct * edge.P * (math.sqrt(parent_N) / (1 + edge.N))

    def select(self, node: Node):
        if node.is_leaf_node():
            return node

        max_utc_child = None
        max_utc_value = -1000000000.
        for edge, child_node in node.child_edge_node:
            utc_val = self.utc_value(edge, edge.parent_node.parent_edge.N)
            val = edge.Q

            if edge.parent_node.board.turn == othello.Player.Black:
                val = -edge.Q

            utc_val_child = val + utc_val

            if utc_val_child > max_utc_value:
                max_utc_child = child_node
                max_utc_value = utc_val_child

        all_best_childs = []
        for edge, child_node in node.child_edge_node:
            utc_val = self.utc_value(edge, edge.parent_node.parent_edge.N)
            val = edge.Q

            if edge.parent_node.board.turn == othello.Player.Black:
                val = -edge.Q

            utc_val_child = val + utc_val
            if utc_val_child == max_utc_child:
                all_best_childs.append(child_node)

        if max_utc_child is None:
            raise ValueError("could not identify child with best uct value")

        if len(all_best_childs) > 1:
            idx = random.randint(0, len(all_best_childs) - 1)
            return self.select(all_best_childs[idx])
        else:
            return self.select(max_utc_child)

    def expand_and_evaluate(self, node: Node):
        outcome = node.board.outcome()
        if outcome is not None:
            v = 0.0
            if outcome.winner == othello.Player.White:
                v = 1.0
            elif outcome.winner == othello.Player.Black:
                v = -1.0
            self.backpropagate(v, node.parent_edge)
            return

        v = node.expand(self.network)
        self.backpropagate(v, node.parent_edge)

    def backpropagate(self, v, edge):
        edge.N += 1
        edge.W = edge.W + v
        edge.Q = edge.W / edge.N
        if edge.parent_node is not None:
            if edge.parent_node.parent_edge is not None:
                self.backpropagate(v, edge.parent_node.parent_edge)

    def search(self, root_node):
        self.root_node = root_node
        _ = self.root_node.expand(self.network)
        for i in range(config.MCTS_NODES):
            selected_node = self.select(root_node)
            self.expand_and_evaluate(selected_node)

        N_sum = 0
        move_props = []
        for edge, _ in root_node.child_edge_node:
            N_sum += edge.N
        for (edge, node) in root_node.child_edge_node:
            prob = (edge.N ** (1 / self.tau)) / (N_sum ** (1 / self.tau))
            move_props.append((edge.move, prob, edge.N, edge.Q))

        return move_props
