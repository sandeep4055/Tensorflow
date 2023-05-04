import tensorflow as tf

if __name__ == "__main__":

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu, [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3200)])

    img_height = 224
    img_width = 224
    batch_size = 8

    # data loading
    directory = r"C:\Users\vamsi\OneDrive\Documents\project1\flowers"

    train_data = tf.keras.utils.image_dataset_from_directory(directory=directory,
                                                             validation_split=0.2,
                                                             subset="training",
                                                             shuffle=True,
                                                             seed=1234,
                                                             image_size=(img_height,img_height),
                                                             batch_size=batch_size)

    val_data = tf.keras.utils.image_dataset_from_directory(directory=directory,
                                                           validation_split=0.2,
                                                           shuffle=True,
                                                           seed=1234,
                                                           image_size=(img_height,img_width),
                                                           batch_size=batch_size,
                                                           subset="validation")

    train = train_data.cache().prefetch(tf.data.AUTOTUNE)
    val = val_data.cache().prefetch(tf.data.AUTOTUNE)

    # data augmentation
    augmentation = tf.keras.Sequential(layers=[
        tf.keras.layers.RandomContrast(factor=0.2),
        tf.keras.layers.RandomRotation(factor=0.2)
    ])

    rescale = tf.keras.layers.Rescaling(scale=1./255)

    # Model

    model = tf.keras.Sequential(layers=[
        tf.keras.layers.Input(shape=(img_height, img_width, 3)),
        rescale,
        augmentation,
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="SAME"),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="SAME"),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="SAME"),
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dense(units=5, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=["accuracy"])

    # callbacks
    earlystopping = tf.keras.callbacks.EarlyStopping(patience=5)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("model1.h5", save_best_only=True)

    # model fit
    # model.fit(train, validation_data=val, epochs=30, callbacks=[earlystopping, checkpoint])

    # Transfer learning
    # We can also use tensorflow hub, but we are using tf.keras.application

    pre_trained = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(img_height, img_width, 3))
    pre_trained.trainable = False
    preprocess_layer = tf.keras.applications.mobilenet_v2.preprocess_input

    """
    # fine tuning
        for layers in pre_trained.layers[:5]:
        layers.trainable = True
    """

    # Model 2

    input_layer = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    augmentation_layer = augmentation(input_layer)
    pre_process_layer = preprocess_layer(augmentation_layer)
    pre_trained_layer = pre_trained(pre_process_layer)
    global_max_pooling_layer = tf.keras.layers.GlobalMaxPool2D()(pre_trained_layer)
    output_layer = tf.keras.layers.Dense(5, activation="softmax")(global_max_pooling_layer)

    model2 = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # model2.summary()

    model2.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.sparse_categorical_crossentropy,
                   metrics=["accuracy"])

    # model2.fit(train, validation_data=val, epochs=30, callbacks=[earlystopping, checkpoint])

    # Efficient net

    efficient_net = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(img_height, img_width, 3))
    efficient_net.trainable = False
    efficient_net_preprocess_layer = tf.keras.applications.efficientnet.preprocess_input

    input_layer = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    augmentation_layer = augmentation(input_layer)
    pre_process_layer = efficient_net_preprocess_layer(augmentation_layer)
    pre_trained_layer = efficient_net(pre_process_layer)
    global_max_pooling_layer = tf.keras.layers.GlobalMaxPool2D()(pre_trained_layer)
    output_layer = tf.keras.layers.Dense(5, activation="softmax")(global_max_pooling_layer)

    model3 = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model3.summary()

    model3.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.sparse_categorical_crossentropy,
                   metrics=["accuracy"])

    model3.fit(train, validation_data=val, epochs=30, callbacks=[earlystopping, checkpoint])
















