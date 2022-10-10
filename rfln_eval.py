import gc
import random

import tensorflow as tf
import numpy as np

import game.othello as othello
import mcts
import rfln_train

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


def rand_vs_net(net_color: othello.Player):
    model = tf.keras.models.load_model("models/nn_v1.keras")
    batch_size = 100
    terminals_num = 0
    moves_num = 0

    white_wins = 0
    black_wins = 0
    draws = 0

    current_nodes = [None] * batch_size
    for i in range(batch_size):
        current_nodes[i] = mcts.Node(mcts.FakeNode(), 0, othello.Player.Black, othello.Board())
        current_nodes[i].is_game_root = True
        current_nodes[i].is_search_root = True

    mcts_searcher = mcts.MCTS(model, 2)

    while terminals_num != batch_size:
        terminals_num = 0
        moves_num += 1
        gc.collect()

        if current_nodes[0].player == net_color:
            pi_batch = mcts_searcher.alpha(current_nodes)

            for i in range(batch_size):
                if current_nodes[i].is_terminal is True:
                    terminals_num += 1
                else:
                    move = np.argmax(pi_batch[i])
                    current_nodes[i] = rfln_train.make_move(current_nodes[i], move)
        else:
            mcts_searcher.alpha(current_nodes)
            for i in range(batch_size):
                if current_nodes[i].board.outcome() is not None:
                    terminals_num += 1
                else:
                    move = random.choice(current_nodes[i].board.legal_moves())
                    current_nodes[i] = rfln_train.make_move(current_nodes[i], move)


    for node in current_nodes:
        white_cnt = bin(node.board.bbs[othello.Player.White]).count('1')
        black_cnt = bin(node.board.bbs[othello.Player.Black]).count('1')
        if white_cnt > black_cnt:
            white_wins += 1
        elif black_cnt > white_cnt:
            black_wins += 1
        else:
            draws += 1

    all = white_wins + black_wins
    print(f"Network vs Rand: whiteWins: {white_wins}, blackWins: {black_wins}, draws: {draws}")
    print("Network vs Rand: " + str(white_wins / all) + "/" + str(black_wins / all))


rand_vs_net(othello.Player.White)
rand_vs_net(othello.Player.Black)
