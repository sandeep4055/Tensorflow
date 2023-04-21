import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    # To use tensorflow data api first convert our data into tensorflow datasets

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = np.arange(0, 1000)

    # data to tf datasets
    dataset_x = tf.data.Dataset.from_tensor_slices(x)
    dataset_y = tf.data.Dataset.from_tensor_slices(y)  # lists or numpy arrays

    print(type(dataset_x), type(dataset_y))

    # indexing of tensor datasets
    for data in dataset_y.take(5):
        print(data)

    # Auto-tune
    auto_tune = tf.data.AUTOTUNE

    # operation on values/instance using map
    dataset_x = dataset_x.map(lambda value: value**2, num_parallel_calls=auto_tune)
    dataset_y = dataset_y.map(lambda value: value**3,num_parallel_calls=auto_tune)

    # for checking no.of instances i.e(=len) in tf dataset we use tf.cardinality
    print(tf.data.Dataset.cardinality(dataset_y))   # also we can use dataset_y.cardinality()

    # cache == cache our data in memory
    # shuffle = shuffle tf dataset data
    # batch == divide tf dataset into batches of particular batch size
    # prefetch == prefetch the next batch

    """for instance in dataset_y.shuffle(buffer_size=100,seed=200).take(5):
        print(instance)"""

    """for instance in dataset_y.batch(batch_size=32,num_parallel_calls=auto_tune).take(1):
        print(instance)"""

    dataset_y = dataset_y.cache().shuffle(buffer_size=100,
                                          seed=21).batch(
                                                    batch_size=32, num_parallel_calls=auto_tune).prefetch(
                                                                                                        auto_tune)
    """for instance in dataset_y.take(1):
        print(instance)"""

    print(dataset_y.cardinality())










