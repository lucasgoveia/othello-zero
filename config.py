ROW_SIZE = 8
COLUMN_SIZE = 8
BOARD_SIZE = ROW_SIZE * COLUMN_SIZE
PASS_MOVE = 64
MOVES_CNT = 65

LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 128

# == Encode pos == 8 x 8 x 5
# 8x8 for white
# 8x8 for black
# 8x8 for empty
# 8x8 for 1 pass move
# 8x8 for 2 pass move
#
# == Last 7 pos ==  8 x 8 x 35
#
# == Turn == 8 x 8 x 1
#
# total = 41 x 8 x 8
