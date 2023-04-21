import tensorflow as tf
import os
import glob
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_files(paths_list):
    images, labels_names = [], []

    for img_path in paths_list:
        images.append(img_path)
        labels_names.append(img_path.split("\\")[-2])

    return images, labels_names


def read_image(image_path, image_label):
    # read image
    image = tf.io.read_file(filename=image_path)
    # decode image
    img = tf.image.decode_jpeg(image)
    img_resized = tf.image.resize(img, size=(256, 256))

    return img_resized, image_label


def normalize(image, label):
    # divide by 255.0
    image = tf.divide(image, 255.0)

    return image, label


if __name__ == "__main__":

    # Definitions
    AUTONE = tf.data.AUTOTUNE
    IMG_SHAPE = (256, 256, 3)
    batch_size = 32
    buffer_size = 4000
    seed = 21

    root_dir = r"C:\Users\vamsi\OneDrive\Documents\Tensorflow\flowers"
    paths = glob.glob(os.path.join(root_dir, "*", "*.jpg"), recursive=True)

    images_path, labels = process_files(paths)

    # Label Encoder
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    # Tensor flow dataset

    dataset = tf.data.Dataset.from_tensor_slices(tensors=(images_path, labels_encoded))

    train_size = int(dataset.cardinality().numpy() * 0.80)
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)

    train = dataset.take(train_size)
    test = dataset.skip(train_size)

    # print(train.cardinality(), test.cardinality())

    # Validation
    train_size = int(train.cardinality().numpy() * 0.80)
    train = train.shuffle(buffer_size=buffer_size, seed=seed)

    validation = train.skip(train_size)
    train = train.take(train_size)

    # Read Image
    train = train.map(read_image, num_parallel_calls=AUTONE)
    test = test.map(read_image, num_parallel_calls=AUTONE)
    validation = validation.map(read_image, num_parallel_calls=AUTONE)

    """# Normalization
    train = train.map(normalize, num_parallel_calls=AUTONE)
    test = test.map(normalize, num_parallel_calls=AUTONE)
    validation = validation.map(normalize, num_parallel_calls=AUTONE)"""

    # Initializing batches

    train = train.cache().\
        shuffle(buffer_size=train.cardinality().numpy(), seed=21).\
        batch(batch_size=batch_size, num_parallel_calls=AUTONE).\
        prefetch(AUTONE)

    test = test.cache(). \
        batch(batch_size=batch_size, num_parallel_calls=AUTONE). \
        prefetch(AUTONE)

    validation = validation.cache(). \
        batch(batch_size=batch_size, num_parallel_calls=AUTONE). \
        prefetch(AUTONE)

    # Data Augmentation

    augmentation = tf.keras.Sequential(layers=
                                       [tf.keras.layers.RandomRotation(factor=0.20),
                                        tf.keras.layers.RandomContrast(factor=0.20)])

    rescaling = tf.keras.layers.Rescaling(1./255.0)

    # Model Creation
    input_layer = tf.keras.layers.Input(shape=IMG_SHAPE)
    flatten_layer = tf.keras.layers.Flatten()(input_layer)
    dense_1 = tf.keras.layers.Dense(units=100, activation="relu")(flatten_layer)
    dense_2 = tf.keras.layers.Dense(units=150, activation="relu")(dense_1)
    dense_3 = tf.keras.layers.Dense(units=200, activation="relu")(dense_2)
    output_layer = tf.keras.layers.Dense(units=5, activation="softmax")(dense_3)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

    history = model.fit(train, validation_data=validation, epochs=5)

    pd.DataFrame(history.history).plot()

    plt.grid()
    plt.show()










