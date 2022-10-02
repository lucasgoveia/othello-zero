from enum import IntEnum

import numpy as np

Bitboard = np.uint64

DEBRUIJ_T = [
    0, 47, 1, 56, 48, 27, 2, 60,
    57, 49, 41, 37, 28, 16, 3, 61,
    54, 58, 35, 52, 50, 42, 21, 44,
    38, 32, 29, 23, 17, 11, 4, 62,
    46, 55, 26, 59, 40, 36, 15, 53,
    34, 51, 20, 43, 31, 22, 10, 45,
    25, 39, 14, 33, 19, 30, 9, 24,
    13, 18, 8, 12, 7, 6, 5, 63
]

DEBRUIJ_M = np.uint64(0x03f79d71b4cb0a89)


def bb_scan_forward(bb: Bitboard):
    return DEBRUIJ_T[((bb ^ (bb - np.uint64(1))) * DEBRUIJ_M) >> np.uint64(58)]


class Player(IntEnum):
    White = 0
    Black = 1

    def __unicode__(self):
        return '⚪' if self == Player.White else '⚫'

    def __str__(self):
        return self.__unicode__()

    def other(self):
        if self == Player.White:
            return Player.Black
        else:
            return Player.White


LEFT_RIGHT_MAKS = np.uint64(0x7e7e7e7e7e7e7e7e)
TOP_BOTTOM_MASK = np.uint64(0x00ffffffffffff00)
CORNER_MASK = LEFT_RIGHT_MAKS & TOP_BOTTOM_MASK

DELTAS = [
    (LEFT_RIGHT_MAKS, lambda bb: bb >> np.uint64(1)),
    (CORNER_MASK, lambda bb: bb >> np.uint64(9)),
    (TOP_BOTTOM_MASK, lambda bb: bb >> np.uint64(8)),
    (CORNER_MASK, lambda bb: bb >> np.uint64(7)),
    (LEFT_RIGHT_MAKS, lambda bb: bb << np.uint64(1)),
    (CORNER_MASK, lambda bb: bb << np.uint64(9)),
    (TOP_BOTTOM_MASK, lambda bb: bb << np.uint64(8)),
    (CORNER_MASK, lambda bb: bb << np.uint64(7)),
]

PASS_MOVE_CODE = 64

MOVE_DISPLAY = ["a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
                "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
                "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
                "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
                "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
                "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
                "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
                "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
                "pm"]


def print_bitboard(bb: Bitboard):
    bb_repr = '   +___+___+___+___+___+___+___+___+\n'
    for rank in range(8):
        for file in range(8):
            sq = rank * 8 + file
            occ = (bb & np.uint64(1 << sq)) > 0

            if file == 0:
                bb_repr += f' {rank + 1} |'

            if occ:
                bb_repr += f'❌ |'
            else:
                bb_repr += f'   |'

        bb_repr += '\n   +___+___+___+___+___+___+___+___+\n'

    bb_repr += '     a   b   c   d   e   f   g   h  \n\n'
    print(bb_repr)


class Move:
    def __init__(self, pos):
        self.pos = pos

    def __str__(self):
        return MOVE_DISPLAY[self.pos]

    def __repr__(self):
        return self.__str__()


class Outcome:
    def __init__(self, winner: Player = None):
        self.winner = winner


class Board:
    def __init__(self):
        white = np.uint64(0x1008000000)
        black = np.uint64(0x810000000)
        self.bbs = [white, black]
        self.turn = Player.Black
        self.pass_move_count = 1
        self.prev = None

    def __unicode__(self):
        board_repr = '   +___+___+___+___+___+___+___+___+\n'
        for rank in range(8):
            for file in range(8):
                sq = rank * 8 + file
                white_occ = (self.bbs[Player.White] & np.uint64(1 << sq)) > 0
                black_occ = (self.bbs[Player.Black] & np.uint64(1 << sq)) > 0

                if file == 0:
                    board_repr += f' {rank + 1} |'

                if white_occ:
                    board_repr += f'{str(Player.White)} |'
                elif black_occ:
                    board_repr += f'{str(Player.Black)} |'
                else:
                    board_repr += f'   |'

            board_repr += '\n   +___+___+___+___+___+___+___+___+\n'

        board_repr += '     a   b   c   d   e   f   g   h  \n\n'
        return board_repr

    def __str__(self):
        return self.__unicode__()

    def us(self):
        return self.bbs[self.turn]

    def them(self):
        return self.bbs[self.turn.other()]

    def empty(self):
        return ~(self.bbs[Player.White] | self.bbs[Player.Black])

    def legal_moves_bb(self) -> Bitboard:
        moves_bb = np.uint64(0)

        us = self.us()
        them = self.them()
        empty = self.empty()

        for delta in DELTAS:
            offset_fn = delta[1]
            moves_bb |= empty & offset_fn(search_contiguous(delta, us, them))

        return moves_bb

    def legal_moves(self):
        moves_bb = self.legal_moves_bb()

        moves = []

        while moves_bb:
            sq = bb_scan_forward(moves_bb)
            moves_bb &= moves_bb - np.uint64(1)
            moves.append(Move(sq))

        if len(moves) == 0:
            moves.append(Move(PASS_MOVE_CODE))

        return moves

    def __copy__(self):
        new = Board()

        new.prev = self.prev
        new.turn = self.turn
        new.bbs = self.bbs
        new.pass_move_count = self.pass_move_count

        return new

    def apply_move(self, mv: Move):
        if mv.pos == PASS_MOVE_CODE:
            copy = self.__copy__()

            self.prev = copy
            self.turn = self.turn.other()
            self.bbs = self.bbs
            self.pass_move_count = self.pass_move_count + 1

            return copy

        us = self.us()
        them = self.them()

        bitmove = np.uint64(1 << mv.pos)
        flipped_stones = np.uint64(0)

        for delta in DELTAS:
            offset_fn = delta[1]
            s = search_contiguous(delta, bitmove, them)
            flipped_stones |= np.uint64(0) if (us & offset_fn(s)) == 0 else s

        us |= flipped_stones | bitmove
        them &= ~flipped_stones

        prev = self.__copy__()

        self.prev = prev
        self.pass_move_count = 0
        self.turn = prev.turn.other()

        white = us if prev.turn == Player.White else them
        black = us if prev.turn == Player.Black else them

        self.bbs = [white, black]

    def outcome(self):
        if self.empty() == np.uint64(0) or self.pass_move_count >= 2:
            white_cnt = bin(self.bbs[Player.White]).count('1')
            black_cnt = bin(self.bbs[Player.Black]).count('1')

            if white_cnt > black_cnt:
                return Outcome(Player.White)
            elif black_cnt > white_cnt:
                return Outcome(Player.Black)
            else:
                return Outcome()

        return None


def search_contiguous(delta, us: Bitboard, them: Bitboard):
    delta_mask = delta[0]
    offset_fn = delta[1]

    w = them & delta_mask
    t = w & offset_fn(us)
    t |= w & offset_fn(t)
    t |= w & offset_fn(t)
    t |= w & offset_fn(t)
    t |= w & offset_fn(t)
    t |= w & offset_fn(t)
    return t
