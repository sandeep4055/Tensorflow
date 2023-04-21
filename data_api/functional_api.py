import tensorflow as tf

if __name__ == "__main__":

    # Functional Api
    # 1) Input
    # 2) Layers
    # 3) Model >> takes input and output layer

    input_layer = tf.keras.layers.Input(shape=(28, 28))
    conv_1_layer = tf.keras.layers.Conv2D()


