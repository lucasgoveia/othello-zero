import time
from collections import deque

import tables
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
import game.othello as othello
import mcts
import nn_util


class SelfPlay:

    def __init__(self, model):
        self.model = model
        self.positions = []
        self.moves_probs = []
        self.values = []

    def play(self):
        board = othello.Board()

        while board.outcome() is None:
            self.positions.append(nn_util.board_to_nn_input(board))

            root_edge = mcts.Edge(None, None)
            root_edge.N = 1
            root_node = mcts.Node(board.__copy__(), root_edge)
            searcher = mcts.MCTS(self.model)

            search_start = time.time()
            search_result = searcher.search(root_node)
            search_end = time.time()

            print(f'Search elapsed {search_end - search_start}s')

            output = np.zeros(config.MOVES_CNT)
            for (move, prob, _, _) in search_result:
                output[move.pos] = prob

            rand_idx = np.random.multinomial(1, output)
            idx = np.where(rand_idx == 1)[0][0]
            next_move = None

            for move, _, _, _ in search_result:
                if move.pos == idx:
                    next_move = move

            self.moves_probs.append(output)
            board.apply_move(next_move)

            if board.turn == othello.Player.White:
                self.values.append(1)
            else:
                self.values.append(-1)

        else:
            outcome = board.outcome()
            if outcome is not None:
                for i in range(0, len(self.moves_probs)):
                    if outcome.winner == othello.Player.Black:
                        self.values[i] = self.values[i] * (-1.0)


if __name__ == '__main__':
    model = tf.keras.models.load_model("models/current_nn.keras")

    pos = deque(maxlen=12_800)
    move_probs = deque(maxlen=12_800)
    values = deque(maxlen=12_800)

    for iter in tqdm(range(1, 101)):
        for i in tqdm(range(0, 8)):
            game = SelfPlay(model)
            game.play()

            pos.extend(game.positions)
            move_probs.extend(game.moves_probs)
            values.extend(game.values)

        np_pos = np.array(pos)
        np_probs = np.array(move_probs)
        np_vals = np.array(values)

        try:
            with tables.open_file(f'data/iter_{iter}.h5', mode='w') as f:
                f.create_earray(f.root, 'values', tables.Int8Atom(), (0, 1))
                f.create_earray(f.root, 'move_probs', tables.UInt8Atom(), (0, 65))
                f.create_earray(f.root, 'positions', tables.UInt8Atom(), (0, 8, 8, 25))

                f.root.values.append(np_vals.astype(np.short))
                f.root.positions.append(np_pos.astype(np.short))
                f.root.move_probs.append(np_probs.astype(np.short))
                print("Saved data")
        except:
            pass

        model.fit(np_pos, [np_probs, np_vals], epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

        model.save(f'models/checkpoints/iter_{iter}.keras')
