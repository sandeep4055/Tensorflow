import tensorflow as tf


if __name__ == "__main__":

    # data from url
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    path_to_downloaded_file = tf.keras.utils.get_file(
        "flower_photos",
        url,
        untar=True)

    print(path_to_downloaded_file)

    train = tf.keras.utils.image_dataset_from_directory(path_to_downloaded_file,
                                                        seed=1234,
                                                        validation_split=0.2,
                                                        subset="training")

    validation = tf.keras.utils.image_dataset_from_directory(path_to_downloaded_file,
                                                             seed=1234,
                                                             validation_split=0.2,
                                                             subset="validation")

    for img , label in train.take(1):
        print(img.shape)
        print(label)



