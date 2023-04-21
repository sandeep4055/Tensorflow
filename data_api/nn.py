import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        train_size = x_train.shape[0]*0.80
        index = range(len(x_train))
        shuffle_index = tf.random.shuffle(index)

        X_train = tf.gather(x_train, shuffle_index[:int(train_size)])
        Y_train = tf.gather(y_train, shuffle_index[:int(train_size)])

        X_val = tf.gather(x_train, shuffle_index[int(train_size):])
        Y_val = tf.gather(y_train, shuffle_index[int(train_size):])

        # print(f"the train_size : {X_train.shape,Y_train.shape}")
        # print(f"the train_size : {X_val.shape,Y_val.shape}")
        # print("-------------------------")

        # data preprocessing
        X_train = tf.cast(X_train, tf.dtypes.float32)
        X_val = tf.cast(X_val, tf.dtypes.float32)
        X_test = tf.cast(x_test, tf.dtypes.float32)

        X_train = tf.divide(X_train, 255.0)
        X_val = tf.divide(X_val, 255.0)
        X_test = tf.divide(X_test, 255.0)

        # model building

        # Sequential Api
        # Functional Api
        # subclassing Api

        """model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)), # similar to np.ravel
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=10, activation="softmax")
            ]
        )"""
        # Functional Api
        input_layer = tf.keras.layers.Input(shape=X_train.shape[1:])
        flatten_layer = tf.keras.layers.Flatten()(input_layer)
        dense_layer = tf.keras.layers.Dense(units=100, activation="relu")(flatten_layer)
        output_layer = tf.keras.layers.Dense(units=10, activation="softmax")(dense_layer)

        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])

        # history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10)

        """for layer in model.layers:
                print(layer.name)"""

        # print(model.layers[2].get_weights())

        # copy only architecture
        # model2 = model >>> not the good way

        model2 = tf.keras.models.clone_model(model)  # better way to copy model

        model2.set_weights(model.get_weights())

        # model2.summary()

        # model2.save("mnist_nn.h5")

        # m = tf.keras.models.load_model("mnist_nn.h5")
        # m.predict()

        # Prediction

        """prediction = model.predict(X_test)
        predict = np.argmax(prediction, axis=1)
        print("------------------------------------")
        print(np.sum(y_test == predict)/len(prediction))"""

        # plot model
        tf.keras.utils.plot_model(model=model,
                                  show_dtype=True,
                                  show_layer_names=True,
                                  show_layer_activations=True)

        """pd.DataFrame(history.history).plot()
        plt.grid()
        plt.show()"""
