import random

import tensorflow as tf
import numpy as np

import game.othello as othello
import mcts
import nn_util

random.seed(10)


def rand_vs_rand(board: othello.Board) -> othello.Outcome:
    while board.outcome() is None:
        moves = list(board.legal_moves())
        m = random.choice(moves)
        board.apply_move(m)

    return board.outcome()


whiteWins = 0
blackWins = 0
draws = 0

for i in range(0, 100):
    board = othello.Board()
    outcome = rand_vs_rand(board)

    if outcome.winner == othello.Player.White:
        whiteWins += 1
    elif outcome.winner == othello.Player.Black:
        blackWins += 1
    else:
        draws += 0

all_wins = whiteWins + blackWins
print("Rand vs Rand: " + str(whiteWins / all_wins) + "/" + str(blackWins / all_wins))
print(f"Rand vs Rand: whiteWins: {whiteWins}, blackWins: {blackWins}, draws: {draws}")

model = tf.keras.models.load_model("models/checkpoints/iter_21.keras")


def best_net_legal_move(board: othello.Board) -> othello.Move:
    root_edge = mcts.Edge(None, None)
    root_edge.N = 1
    root_node = mcts.Node(board.__copy__(), root_edge)

    searcher = mcts.MCTS(model)
    search_result = searcher.search(root_node)

    best_mv = othello.Move(65)
    best_prob = 0

    for (move, prob, _, _) in search_result:
        if prob > best_prob:
            best_prob = prob
            best_mv = move

    return best_mv


def rand_vs_net(board: othello.Board, net_color: othello.Player) -> othello.Outcome:
    while board.outcome() is None:
        if board.turn == net_color:
            m = best_net_legal_move(board)
            board.apply_move(m)
            print(board)
        else:
            moves = list(board.legal_moves())
            m = random.choice(moves)
            board.apply_move(m)

    return board.outcome()


whiteWins = 0
blackWins = 0
draws = 0

for i in range(0, 100):
    board = othello.Board()
    print(F'==== GAME {i + 1} ====')
    outcome = rand_vs_net(board, net_color=othello.Player.White)

    if outcome.winner == othello.Player.White:
        whiteWins += 1
    elif outcome.winner == othello.Player.Black:
        blackWins += 1
    else:
        draws += 1

    all = whiteWins + blackWins
    print(f"Network vs Rand: whiteWins: {whiteWins}, blackWins: {blackWins}, draws: {draws}")

# print("== Network as WHITE ==")
# print("Network vs Rand: " + str(whiteWins / all) + "/" + str(blackWins / all))
# print(f"Network vs Rand: whiteWins: {whiteWins}, blackWins: {blackWins}, draws: {draws}")