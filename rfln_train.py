import gc
import os
import random
import time
from multiprocessing import Pool, Process
from pathlib import Path

import numpy as np
import tables
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


def validate(board, move):
    legal_moves = board.legal_moves()
    if move not in legal_moves:
        raise ValueError(f"Invalid move expected any of {legal_moves}, got {move}")


def make_move(node, move):
    validate(node.board, move)
    if move not in node.child_nodes:
        node = mcts.Node(node, move, node.player.other())
    else:
        node = node.child_nodes[move]
    node.is_search_root = True
    node.parent.child_nodes.clear()
    node.parent.is_search_root = False
    return node


class SelfPlayWorker:

    def __init__(self, version, worker_id, batch_size=256):
        self.version = version
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.fake_nodes = [None] * batch_size
        self.current_nodes = [None] * batch_size

    def start(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        model = tf.keras.models.load_model(f"models/nn_v{self.version}.keras")
        mcts_searcher = mcts.MCTS(model, config.MCTS_NODES)

        print("selfplay worker", self.worker_id, "version:", self.version, "echo:", "session start.")
        self.play(mcts_searcher)
        self.save()
        print("selfplay worker", self.worker_id, "session end.")

    def play(self, mcts_searcher: mcts.MCTS):
        terminals_num = 0
        moves_num = 0

        for i in range(self.batch_size):
            self.fake_nodes[i] = mcts.FakeNode()
            self.current_nodes[i] = mcts.Node(self.fake_nodes[i], 0, othello.Player.Black, othello.Board())
            self.current_nodes[i].is_game_root = True
            self.current_nodes[i].is_search_root = True

        while terminals_num != self.batch_size:
            terminals_num = 0
            moves_num += 1
            gc.collect()
            search_start = time.time()
            pi_batch = mcts_searcher.alpha(self.current_nodes)
            search_end = time.time()
            print(f"{moves_num} - Searched move... elapsed: {search_end - search_start}")

            if self.worker_id == 0:
                print(self.current_nodes[0].board)

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

        Path(f'data/v{self.version}/').mkdir(parents=True, exist_ok=True)
        table_file_name = f'data/v{self.version}/{self.batch_size}_{self.worker_id}'

        with tables.open_file(table_file_name, mode='w') as f_out:
            f_out.create_earray(f_out.root, 'values', tables.Float32Atom(), (0,))
            f_out.create_earray(f_out.root, 'move_probs', tables.Float32Atom(), (0, config.MOVES_CNT))
            f_out.create_earray(f_out.root, 'positions', tables.Float32Atom(), (0, 8, 8, config.INPUT_PLANES_CNT))

        for node in self.current_nodes:
            winner = 0
            white_cnt = bin(node.board.bbs[othello.Player.White]).count('1')
            black_cnt = bin(node.board.bbs[othello.Player.Black]).count('1')
            if white_cnt > black_cnt:
                winner = 1.0
            elif black_cnt > white_cnt:
                winner = -1.0

            current = node
            while True:
                pos.append(current.to_features())
                move_probs.append(current.pi)
                values.append(winner)
                if current.is_game_root:
                    break
                current = current.parent

        with tables.open_file(table_file_name, mode='a') as f:
            f.root.positions.append(np.array(pos))
            f.root.move_probs.append(np.array(move_probs))
            f.root.values.append(np.array(values))


def self_play_worker(version, worker_id):
    game = SelfPlayWorker(version, worker_id)
    game.start()


def train_worker(version):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    log_dir = f"logs/fit/v{version}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    tensorboard = TensorBoard(
        log_dir=os.path.join(os.getcwd(), log_dir),
        histogram_freq=1,
        write_images=True,
        update_freq='epoch'
    )

    version_data_dir = f'data/v{version}/'
    data_files = [os.path.join(version_data_dir, f) for f in os.listdir(version_data_dir) if
                  os.path.isfile(os.path.join(version_data_dir, f))]

    positions = np.zeros((0, 8, 8, config.INPUT_PLANES_CNT), dtype=float)
    move_probs = np.zeros((0, config.MOVES_CNT), dtype=float)
    values = np.zeros((0,), dtype=float)

    for tb_filename in data_files:
        with tables.open_file(tb_filename, mode='r') as f:
            move_probs = np.concatenate((move_probs, np.array(f.root.move_probs)))
            positions = np.concatenate((positions, np.array(f.root.positions)))
            values = np.concatenate((values, np.array(f.root.values)))

    model = tf.keras.models.load_model(f"models/nn_v{version}.keras")
    model.fit(
        positions, [move_probs, values],
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stop, tensorboard]
    )

    model.save(f'models/nn_v{version + 1}.keras')


if __name__ == '__main__':
    for e in range(100):
        pool = Pool(5)

        for i in range(5):
            pool.apply_async(self_play_worker, (e, i))

        pool.close()
        pool.join()

        train_process = Process(target=train_worker, args=(e,))
        train_process.start()
        train_process.join()
