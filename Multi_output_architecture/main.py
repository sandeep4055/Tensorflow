import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def preprocess(dataset):

    y1 = dataset.pop("Y1")
    y2 = dataset.pop("Y2")

    y1 = np.array(y1)
    y2 = np.array(y2)

    return y1, y2

# normalize funcction
def normalize(dataset):
    normalized_df = (dataset - dataset.mean()) / dataset.std()
    return normalized_df


if __name__ == "__main__":

    data = pd.read_excel("ENB2012_data.xlsx")

    print(data.head())
    print(data.shape)

    train, test = train_test_split(data, test_size=0.15, random_state=42)

    train, val = train_test_split(train, test_size=0.15, random_state=42)

    train_y = preprocess(train)
    val_y = preprocess(val)
    test_y = preprocess(test)

    train_x = normalize(train)
    val_x = normalize(val)
    test_x = normalize(test)

    # Model Creation

    input_layer = tf.keras.layers.Input(shape=len(train.columns))
    dense_1 = tf.keras.layers.Dense(units=64, activation="relu")(input_layer)
    dense_2 = tf.keras.layers.Dense(units=128, activation="relu")(dense_1)

    output_layer_1 = tf.keras.layers.Dense(units=1, name="y1_output")(dense_2)  #no activation because of regression problem

    dense_3 = tf.keras.layers.Dense(units=256, activation="relu")(dense_2)

    output_layer_2 = tf.keras.layers.Dense(units=1, name="y2_output")(dense_3)

    model = tf.keras.Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss={
                      "y1_output": "mse",
                      "y2_output": "mse"
                  },
                  metrics={
                      "y1_output": tf.keras.metrics.RootMeanSquaredError(),
                      "y2_output": tf.keras.metrics.RootMeanSquaredError()
                  })

    model.summary()

    """tf.keras.utils.plot_model(model,
                              to_file="model.png",
                              show_dtype=True,
                              show_shapes=True,
                              show_layer_names=True,
                              show_layer_activations=True)
"""

    history = model.fit(x=train_x, y=train_y, epochs=500, validation_data=(val_x, val_y), batch_size=16)

    loss, y1_loss, y2_loss, y1_rmse, y2_rmse = model.evaluate(x=test_x, y=test_y)

    plt.plot(pd.DataFrame(history.history))









