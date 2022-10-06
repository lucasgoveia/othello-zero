import random

import nn_util
from game.othello import Board

board = Board()

while board.pass_move_count < 2:
    print(board)

    moves = board.legal_moves()
    print(moves)

    mv = random.choice(moves)
    board.apply_move(mv)

    test = nn_util.board_to_nn_input(board)
    test = test.transpose()
    print(test)
