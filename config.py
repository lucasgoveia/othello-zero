ROW_SIZE = 8
COLUMN_SIZE = 8
BOARD_SIZE = ROW_SIZE * COLUMN_SIZE
PASS_MOVE = 64
MOVES_CNT = 65

LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 128
INPUT_PLANES_CNT = 25

# == Encode pos == 8 x 8 x 3
# 8x8 for white
# 8x8 for black
# 8x8 for empty

#
# == Last 7 pos ==  8 x 8 x 21
#
# == Turn == 8 x 8 x 1
#
# total = 25 x 8 x 8
