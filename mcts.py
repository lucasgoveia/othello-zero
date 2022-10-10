import math

import numpy as np

import config
import game.othello as othello
import nn_util


def normalize_with_mask(x, mask):
    x_masked = np.multiply(x, mask)
    x_normalized = x_masked / np.sum(x_masked)
    return x_normalized


def player_val(player: othello.Player):
    if player == othello.Player.White:
        return 1.0
    else:
        return -1.0


class FakeNode:
    def __init__(self):
        self.parent = None
        self.edge_N = np.zeros([config.MOVES_CNT], dtype=np.float)
        self.edge_W = np.zeros([config.MOVES_CNT], dtype=np.float)


class Node:
    def __init__(self, parent, move: othello.Move, player: othello.Player, board: othello.Board = None):
        self.parent = parent
        self.expanded = False
        self.move = move
        self.player = player

        new_board = board if board is not None else self.parent.board.__copy__()
        if board is None:
            new_board.apply_move(move)

        self.board = new_board
        self.legal_moves = nn_util.encode_moves_bb(self.board.legal_moves_bb())
        self.child_nodes = {}
        self.is_game_root = False
        self.is_search_root = False
        self.is_terminal = False
        self.pi = np.zeros([config.MOVES_CNT], dtype=np.float)
        self.edge_N = np.zeros([config.MOVES_CNT], dtype=np.float)
        self.edge_W = np.zeros([config.MOVES_CNT], dtype=np.float)
        self.edge_P = np.zeros([config.MOVES_CNT], dtype=np.float)

    @property
    def edge_Q(self):
        return self.edge_W / (self.edge_N + (self.edge_N == 0))

    @property
    def self_N(self):
        return self.parent.edge_N[self.move]

    @self_N.setter
    def self_N(self, n):
        self.parent.edge_N[self.move] = n

    @property
    def edge_U(self):
        return config.C_PUCT * self.edge_P * math.sqrt(max(1, self.self_N)) / (1 + self.edge_N)

    @property
    def edge_U_with_noise(self):
        noise = normalize_with_mask(np.random.dirichlet([config.NOISE_ALPHA] * config.MOVES_CNT), self.legal_moves)
        p_with_noise = self.edge_P * (1 - config.NOISE_WEIGHT) + noise * config.NOISE_WEIGHT
        return config.C_PUCT * p_with_noise * math.sqrt(max(1, self.self_N)) / (1 + self.edge_N)

    @property
    def edge_Q_plus_U(self):
        if self.is_game_root:
            return self.edge_Q * player_val(self.player) + self.edge_U_with_noise + self.legal_moves * 1000
        else:
            return self.edge_Q * player_val(self.player) + self.edge_U + self.legal_moves * 1000

    @property
    def self_W(self):
        return self.parent.edge_W[self.move]

    @self_W.setter
    def self_W(self, w):
        self.parent.edge_W[self.move] = w

    def to_features(self):
        features = np.zeros([8, 8, config.INPUT_PLANES_CNT]).astype(np.float)
        current = self
        for i in range(config.HISTORY_CNT + 1):
            nn_util.encode_bitboard(current.board.bbs[othello.Player.White], features, 2 * i)
            nn_util.encode_bitboard(current.board.bbs[othello.Player.Black], features, 2 * i + 1)
            if current.is_game_root:
                break
            current = current.parent

        if self.board.turn == othello.Player.White:
            features[:, :, 12] = 1

        return features


class MCTS:
    def __init__(self, nn, nodes_cnt):
        self.nn = nn
        self.nodes_cnt = nodes_cnt

    def select(self, nodes):
        best_nodes_batch = [None] * len(nodes)
        for i, node in enumerate(nodes):
            current = node
            while current.expanded:
                best_edge = np.argmax(current.edge_Q_plus_U)
                if best_edge not in current.child_nodes:
                    current.child_nodes[best_edge] = Node(current, best_edge, current.player.other())
                if current.is_terminal:
                    break
                if current.board.outcome() is not None:
                    current.is_terminal = True
                    break

                current = current.child_nodes[best_edge]

            best_nodes_batch[i] = current
        return best_nodes_batch

    def expand_and_evaluate(self, nodes_batch: [Node]):
        features_batch = np.zeros((len(nodes_batch), 8, 8, 13), dtype=np.float)
        for i, node in enumerate(nodes_batch):
            node.expanded = True
            features_batch[i] = node.to_features()

        qs = self.nn.predict(features_batch, batch_size=len(nodes_batch))
        p_batch = qs[0]
        v_batch = qs[1]

        for i, node in enumerate(nodes_batch):
            node.edge_P = normalize_with_mask(p_batch[i], node.legal_moves)

        return v_batch

    def backpropagate(self, nodes_batch, v_batch):
        for i, node in enumerate(nodes_batch):
            current = node
            while True:
                current.self_N += 1
                current.self_W += v_batch[i]
                if current.is_search_root:
                    break
                current = current.parent

    def search(self, nodes):
        best_nodes_batch = self.select(nodes)
        v_batch = self.expand_and_evaluate(best_nodes_batch)
        self.backpropagate(best_nodes_batch, v_batch)

    def alpha(self, nodes):
        for i in range(self.nodes_cnt):
            self.search(nodes)

        pi_batch = np.zeros([len(nodes), config.MOVES_CNT], dtype=np.float)
        for i, node in enumerate(nodes):
            n_with_temperature = node.edge_N ** (1 / config.TAU)
            sum_n_with_temperature = np.sum(n_with_temperature)
            if sum_n_with_temperature == 0:
                node.pi = np.zeros([config.MOVES_CNT], dtype=np.float)
                node.pi[othello.PASS_MOVE] = 1
            else:
                node.pi = n_with_temperature / sum_n_with_temperature
            pi_batch[i] = node.pi
        return pi_batch
