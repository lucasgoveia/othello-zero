import numpy as np

import config
import game.othello as othello


def encode_moves_bb(moves_bb: othello.Bitboard):
    arr = np.zeros(65)

    if moves_bb == np.int64(0):
        arr[64] = 1.0
        return arr

    for sq in range(64):
        if (moves_bb & np.uint64(1 << sq)) != 0:
            arr[sq] = 1.0

    return arr


def encode_bitboard(bitboard: othello.Bitboard, encoded, pos):
    for sq in range(64):
        if (bitboard & np.uint64(1 << sq)) != 0:
            encoded[sq % 8, sq // 8, pos] = 1
