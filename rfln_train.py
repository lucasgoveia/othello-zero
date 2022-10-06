import gc
import os
import random
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, Process
import time

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard

import config
import game.othello as othello
import mcts


def pick_move_probabilistically(pi):
    r = random.random()
    s = 0
    for move in range(len(pi)):
        s += pi[move]
        if s >= r:
            return move
    return np.argmax(pi)


def validate(move):
    if not (isinstance(move, int) or isinstance(move, np.int64)) or not (0 <= move < 64 or move == othello.PASS_MOVE):
        raise ValueError("move must be integer from [0, 63] or {}, got {}".format(othello.PASS_MOVE, move))


def make_move(node, move):
    validate(move)
    if move not in node.child_nodes:
        node = mcts.Node(node, move, -node.player)
    else:
        node = node.child_nodes[move]
    node.is_search_root = True
    node.parent.child_nodes.clear()
    node.parent.is_search_root = False
    return node


class SelfPlayWorker:

    def __init__(self, version, worker_id, batch_size=128):
        self.version = version
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.fake_nodes = [None] * batch_size
        self.current_nodes = [None] * batch_size

    def start(self):
        model = tf.keras.models.load_model(f"models/nn_v{self.version}.keras")
        mcts_searcher = mcts.MCTS(model)

        print("selfplay worker", self.worker_id, "version:", self.version, "echo:", "session start.")
        self.play(mcts_searcher)
        self.save()
        print("selfplay worker", self.worker_id, "session end.")

    def play(self, mcts_searcher: mcts.MCTS):
        terminals_num = 0
        moves_num = 0

        for i in range(self.batch_size):
            self.fake_nodes[i] = mcts.FakeNode()
            self.current_nodes[i] = mcts.Node(self.fake_nodes[i], othello.Move(0), othello.Board())
            self.current_nodes[i].is_game_root = True
            self.current_nodes[i].is_search_root = True

        while terminals_num != self.batch_size:
            terminals_num = 0
            moves_num += 1
            gc.collect()
            search_start = time.time()
            pi_batch = mcts_searcher.alpha(self.current_nodes)
            search_end = time.time()
            print(f"Searched move... elapsed: {search_end - search_start}s")

            for i in range(self.batch_size):
                if self.current_nodes[i].is_terminal is True:
                    terminals_num += 1
                else:
                    move = pick_move_probabilistically(pi_batch[i])
                    self.current_nodes[i] = make_move(self.current_nodes[i], move)

    def save(self):
        pos = []
        move_probs = []
        values = []
        for node in self.current_nodes:
            winner = 0
            if node.board.outcome().winner == othello.Player.White:
                winner = 1.0
            elif node.board.outcome().winner == othello.Player.Black:
                winner = -1.0

            current = node
            while True:
                pos.append(node.to_features())
                move_probs.append(current.pi)
                values.append(winner)
                if current.is_game_root:
                    break
                current = current.parent

        Path(f'data/v{self.version}/pos/').mkdir(parents=True, exist_ok=True)
        Path(f'data/v{self.version}/move_probs/').mkdir(parents=True, exist_ok=True)
        Path(f'data/v{self.version}/values/').mkdir(parents=True, exist_ok=True)

        np.save(f'data/v{self.version}/pos/{self.batch_size}_{self.worker_id}', pos)
        np.save(f'data/v{self.version}/move_probs/{self.batch_size}_{self.worker_id}', move_probs)
        np.save(f'data/v{self.version}/values/{self.batch_size}_{self.worker_id}', values)


def self_play_worker(version, worker_id):
    game = SelfPlayWorker(version, worker_id)
    game.start()


if __name__ == '__main__':
    # Load Tensorboard callback
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(
        log_dir=os.path.join(os.getcwd(), log_dir),
        histogram_freq=1,
        write_images=True,
        update_freq='epoch'
    )

    for e in range(100):
        pool = Pool(8)

        for i in range(8):
            pool.apply_async(self_play_worker, (e, i,))

        pool.close()
        pool.join()

        pos_dir = f'data/v{e}/pos/'
        pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, f))]

        move_probs_dir = f'data/v{e}/move_probs/'
        move_probs_files = [os.path.join(move_probs_dir, f) for f in os.listdir(move_probs_dir) if
                            os.path.isfile(os.path.join(move_probs_dir, f))]

        values_dir = f'data/v{e}/values/'
        values_files = [os.path.join(values_dir, f) for f in os.listdir(values_dir) if
                        os.path.isfile(os.path.join(values_dir, f))]

        pos = np.zeros((0, 8, 8, config.INPUT_PLANES_CNT), dtype=np.float)
        move_probs = np.zeros((0, config.MOVES_CNT), dtype=np.float)
        values = np.zeros((0,), dtype=np.float)

        for filename in pos_files:
            x = np.load(filename)
            pos = np.concatenate((pos, x))

        for filename in move_probs_files:
            x = np.load(filename)
            move_probs = np.concatenate((move_probs, x))

        for filename in values_files:
            x = np.load(filename)
            values = np.concatenate((values, x))

        model = tf.keras.models.load_model(f"models/nn_v{e}.keras")
        model.fit(pos, [move_probs, values], epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, callbacks=[tensorboard])

        model.save(f'models/nn_v{e + 1}.keras')
