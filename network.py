from keras import Model
from keras.layers import *
import tensorflow as tf

import config


def conv2d_block(input_layer: Layer) -> Layer:
    cnv2d = Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_first',
        use_bias=False
    )(input_layer)
    bn = BatchNormalization()(cnv2d)
    return ReLU(name='cnv2d')(bn)


def res2d_block(layer: Layer, i) -> Layer:
    cnv2d_1 = Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_first',
        use_bias=False
    )(layer)
    bn_1 = BatchNormalization()(cnv2d_1)
    relu_1 = ReLU()(bn_1)
    cnv2d_2 = Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_first',
        use_bias=False,
    )(relu_1)
    bn_2 = BatchNormalization()(cnv2d_2)

    return ReLU(name=f'res2d_{i}')(layer + bn_2)


def out_block(layer: Layer) -> (Layer, Layer):
    policy_conv = Conv2D(
        filters=2,
        kernel_size=1,
        strides=1,
        padding='same',
        data_format='channels_first',
        use_bias=False
    )(layer)
    policy_bn = BatchNormalization()(policy_conv)
    policy_relu = ReLU()(policy_bn)
    policy_flatten = Flatten()(policy_relu)
    policy_out = Dense(config.MOVES_CNT, name='policyHead', activation='softmax')(policy_flatten)

    value_conv = Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding='same',
        data_format='channels_first',
        use_bias=False,
    )(layer)
    value_bn = BatchNormalization()(value_conv)
    value_relu = ReLU()(value_bn)
    value_flatten = Flatten()(value_relu)
    value_out1 = Dense(64, activation='relu')(value_flatten)
    value_out2 = Dense(1, activation='tanh', name='valueHead')(Flatten()(value_out1))
    # Activation(activation='tanh')(value_out2)

    return policy_out, value_out2


def get_network():
    input_layer = Input((8, 8, config.INPUT_PLANES_CNT), dtype='float32')

    # block 1 - Conv2d
    conv_layer = conv2d_block(input_layer)

    res_layer = conv_layer
    for i in range(0, 9):
        res_layer = res2d_block(res_layer, i)

    policy, value = out_block(res_layer)

    return Model(input_layer, [policy, value])


if __name__ == "__main__":
    model = get_network()

    policy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    value_loss = 'mean_squared_error'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=['categorical_crossentropy', 'MSE'],
    )
    model.save('models/random_model.keras')
    print(model.summary())
