import tensorflow as tf

if __name__ == "__main__" :

    directory = r"C:\Users\vamsi\OneDrive\Documents\Tensorflow\flowers"

    train_data = tf.keras.utils.image_dataset_from_directory(directory=directory,
                                                        shuffle=True,
                                                        seed=1234,
                                                        validation_split=0.2,
                                                        subset="training")

    val_data = tf.keras.utils.image_dataset_from_directory(directory=directory,
                                                      shuffle=True,
                                                      seed=1234,
                                                      validation_split=0.2,
                                                      subset="validation")


    # print(tf.config.list_physical_devices('GPU'))
