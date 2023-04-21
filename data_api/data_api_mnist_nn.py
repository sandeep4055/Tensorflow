import tensorflow as tf

if __name__ == "__main__":
    AUTOTUNE = tf.data.AUTOTUNE
    buffer_size = 100
    seed = 21
    batch_size = 64
    input_shape = (28, 28)

    # Step 1: Load data

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Step 2: Convert our data to tensorflow dataset

    train = tf.data.Dataset.from_tensor_slices(tensors=(x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices(tensors=(x_test, y_test))

    """for img , label in x_train.take(1):
        print(img, label)"""

    # Step 3: train , validation split

    train_size = int(train.cardinality().numpy() * 0.80)
    train_shuffle = train.shuffle(buffer_size=100, seed=seed)
    train = train_shuffle.take(train_size)
    validation = train_shuffle.skip(train_size)  # skip skips the train size

    print(train.cardinality(), validation.cardinality())

    # Step 4: Normalize values

    def normalize(image, label):
        img = tf.cast(image, dtype=tf.dtypes.float32)
        label = tf.cast(label, dtype=tf.dtypes.float32)

        return tf.divide(img, 255.0), label


    train = train.map(normalize, num_parallel_calls=AUTOTUNE)
    validation = validation.map(normalize, num_parallel_calls=AUTOTUNE)
    test = test.map(normalize, num_parallel_calls=AUTOTUNE)

    # Step 5: Batch Dataset

    train = train.cache(). \
        shuffle(buffer_size=buffer_size, seed=seed). \
        batch(batch_size=32, num_parallel_calls=AUTOTUNE). \
        prefetch(AUTOTUNE)

    validation = validation.cache(). \
        batch(batch_size=32, num_parallel_calls=AUTOTUNE). \
        prefetch(AUTOTUNE)

    test = test.cache(). \
        batch(batch_size=32, num_parallel_calls=AUTOTUNE). \
        prefetch(AUTOTUNE)

    # Step 6 : Model building , training

    input_layer = tf.keras.layers.Input(shape=input_shape)
    flatten_layer = tf.keras.layers.Flatten()(input_layer)
    dense_1 = tf.keras.layers.Dense(units=50, activation="relu")(flatten_layer)
    dense_2 = tf.keras.layers.Dense(units=50, activation="relu")(dense_1)
    output_layer = tf.keras.layers.Dense(units=10, activation="softmax")(dense_2)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=["accuracy"])

    model.fit(train, validation_data=validation, epochs=10)
