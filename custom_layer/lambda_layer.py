import tensorflow as tf

# data loading

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# tf dataset

train = tf.data.Dataset.from_tensor_slices(tensors=(x_train, y_train))

train = train.batch(batch_size=32)

# Model

# Custom function for lambda layer


def custom_lambda_layer_function(x):

    return tf.abs(x)


model = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Lambda(lambda x: tf.abs(x)),
        tf.keras.layers.Dense(units=64),
        tf.keras.layers.Lambda(custom_lambda_layer_function),
        tf.keras.layers.Dense(units=10, activation="softmax")
    ]
)

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.fit(train, epochs=10)
