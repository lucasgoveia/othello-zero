ROW_SIZE = 8
COLUMN_SIZE = 8
BOARD_SIZE = ROW_SIZE * COLUMN_SIZE
PASS_MOVE = 64
MOVES_CNT = 65

LEARNING_RATE = 3e-2
EPOCHS = 100
BATCH_SIZE = 128
INPUT_PLANES_CNT = 13
HISTORY_CNT = 5
RESIDUAL_BLOCKS_CNT = 9


MCTS_NODES = 200
C_PUCT = 1.
TAU = 1.
NOISE_ALPHA = 0.5
NOISE_WEIGHT = 0.25
# == Encode pos == 8 x 8 x 2
# 8x8 for white
# 8x8 for black

#
# == Last 5 pos ==  8 x 8 x 10
#
# == Turn == 8 x 8 x 1
#
# total = 8 x 8 x 13
