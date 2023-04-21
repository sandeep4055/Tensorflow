import tensorflow as tf
import numpy as np

if __name__ == "__main__" :

    directory = r"C:\Users\vamsi\OneDrive\Documents\Tensorflow\flowers"

    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0,
                                                                     featurewise_center=True,
                                                                     featurewise_std_normalization=True,
                                                                     horizontal_flip=True,
                                                                     zoom_range=[0.8, 0.9],
                                                                     brightness_range=[0.1, 0.4],
                                                                     validation_split=0.2,
                                                                     dtype=np.float32)

    train_generator = data_generator.flow_from_directory(directory,
                                                         target_size=(256, 256),
                                                         class_mode="sparse",
                                                         shuffle=True,
                                                         subset="training",
                                                         seed=1234,
                                                         batch_size=32)

    val_generator = data_generator.flow_from_directory(directory,
                                                       target_size=(256, 256),
                                                       class_mode="sparse",
                                                       shuffle=True,
                                                       subset='validation',
                                                       seed=1234,
                                                       batch_size=32)

    model = tf.keras.Sequential(layers=[
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(256, 256, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dense(units=5, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

    #steps_per_epoch = len(train_data)/batch_size
    model.fit(train_generator, validation_data=val_generator, epochs=5, steps_per_epoch=25)

