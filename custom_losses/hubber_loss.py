from abc import ABC

import tensorflow as tf


class My_Huber_Loss(tf.keras.losses.Loss):

    threshold = 1

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def call(self, y_true, y_pred):

        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 *self.threshold))

        # If the is_small_error is true it returns the small_error_loss else returns big_error_loss

        return tf.where(is_small_error, small_error_loss, big_error_loss)


model = tf.keras.Sequential(layers=[

    tf.keras.layers.Dense(units=1, input_shape=[1])

])

model.compile(loss=My_Huber_Loss(1), optimizer="sgd")

history = model.fit(x=[10], y=[1])

print(history.history["loss"])
