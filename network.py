import numpy as np
import tensorflow as tf

import config


def get_network():
    input = tf.keras.layers.Input(shape=(8, 8, config.INPUT_PLANES_CNT), dtype=np.float32)

    l2const = 1e-4
    layer = input
    layer = tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2const))(
        layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Activation("relu")(layer)
    for _ in range(config.RESIDUAL_BLOCKS_CNT):
        res = layer
        layer = tf.keras.layers.Conv2D(128, (3, 3), padding="same",
                                       kernel_regularizer=tf.keras.regularizers.l2(l2const))(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation("relu")(layer)
        layer = tf.keras.layers.Conv2D(128, (3, 3), padding="same",
                                       kernel_regularizer=tf.keras.regularizers.l2(l2const))(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Add()([layer, res])
        layer = tf.keras.layers.Activation("relu")(layer)

    vhead = layer
    vhead = tf.keras.layers.Conv2D(1, (1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2const))(vhead)
    vhead = tf.keras.layers.BatchNormalization()(vhead)
    vhead = tf.keras.layers.Activation("relu")(vhead)
    vhead = tf.keras.layers.Flatten()(vhead)
    vhead = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(l2const))(vhead)
    vhead = tf.keras.layers.Activation("relu")(vhead)
    vhead = tf.keras.layers.Dense(1)(vhead)
    vhead = tf.keras.layers.Activation("tanh", name="vh")(vhead)

    phead = layer
    phead = tf.keras.layers.Conv2D(2, (1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2const))(phead)
    phead = tf.keras.layers.BatchNormalization()(phead)
    phead = tf.keras.layers.Activation("relu")(phead)
    phead = tf.keras.layers.Flatten()(phead)
    phead = tf.keras.layers.Dense(config.MOVES_CNT)(phead)
    phead = tf.keras.layers.Activation("softmax", name="ph")(phead)

    model = tf.keras.models.Model(inputs=[input], outputs=[phead, vhead])
    return model


if __name__ == "__main__":
    model = get_network()

    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(),
        loss=[tf.keras.losses.categorical_crossentropy, tf.keras.losses.mean_squared_error],
        loss_weights=[0.5, 0.5],
        metrics=["accuracy"]
    )
    model.save('models/nn_v0.keras')
    print(model.summary())
