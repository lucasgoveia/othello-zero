import numpy as np

import config
import game.othello as othello


def _encode_bitboard(bitboard: othello.Bitboard, encoded, pos):
    for sq in range(64):
        if (bitboard & np.uint64(1 << sq)) != 0:
            encoded[sq % 8, sq // 8, pos] = 1


def _encode_pos(board: othello.Board, encoded: np.ndarray, index: int):
    _encode_bitboard(board.bbs[othello.Player.White], encoded, index + 0)
    _encode_bitboard(board.bbs[othello.Player.Black], encoded, index + 1)
    _encode_bitboard(board.empty(), encoded, index + 2)


def _encode_last7(board: othello.Board, encoded: np.ndarray):
    prev_seen_cnt = 0
    current_board = board.prev
    while current_board is not None and prev_seen_cnt < 7:
        index = (prev_seen_cnt + 1) * 3
        _encode_pos(current_board, encoded, index)
        current_board = current_board.prev
        prev_seen_cnt += 1


def board_to_nn_input(board: othello.Board) -> np.ndarray:
    encoded_pos = np.zeros([8, 8, config.INPUT_PLANES_CNT]).astype(np.ushort)

    _encode_pos(board, encoded_pos, 0)
    _encode_last7(board, encoded_pos)

    if board.turn == othello.Player.White:
        encoded_pos[:, :, 24] = 1

    return encoded_pos
