# Distribution Strategies

## Table of Contents

- [ What is Distribution Strategies?](#what-is-distribution-strategies)
- [Synchronous training vs Asynchronous training](#synchronous-training-vs-asynchronous-training)
- [Distribution Strategies Types](#here-are-some-distribution-strategies-available-in-tensorflow)

## What is Distribution Strategies?
In TensorFlow, distribution strategies are used to train models across multiple devices or machines, enabling parallelism and scalability. They allow you to take advantage of multiple GPUs or multiple machines to accelerate training and improve performance. 

##### To make sure that we are all on the same page let's define some basic notations:

- **Worker:** a separate machine that contains a CPU and one or more GPUs

- **Accelerator:** a single GPU (or TPU)

- **All-reduce:** a distributed algorithm that aggregates all the trainable parameters from different workers or accelerators. I’m not gonna go into details on how it works but essentially, it receives the weights from all workers and performs some sort of aggregation on them to compute the final weights.

## Synchronous training vs Asynchronous training

Synchronous training and asynchronous training are two different approaches to distributed training in machine learning. Here's an overview of each:

- **Synchronous Training:** In synchronous training, all workers or devices (such as GPUs or machines) are synchronized at each training step. This means that gradients computed by each worker are averaged or aggregated before updating the model parameters. The synchronization step ensures that all workers have the same model weights before proceeding to the next training step. Synchronous training is commonly used in distributed settings with strategies like MirroredStrategy and MultiWorkerMirroredStrategy in TensorFlow. It provides global convergence guarantees and allows for easy implementation of distributed algorithms. However, it may introduce some communication overhead and can be slower if there is significant variability in the computation time across workers.

- **Asynchronous Training:** In asynchronous training, workers update the model parameters independently without waiting for other workers. Each worker computes gradients on its own subset of the data and applies updates to the model asynchronously. Asynchronous training can be faster than synchronous training since workers are not required to wait for each other. However, it can lead to inconsistencies in the model parameters across workers, as different workers may have different versions of the model at any given time. Asynchronous training is more challenging to implement and may require additional techniques, such as controlling the learning rate or using parameter servers, to mitigate the parameter inconsistency issue.

The choice between synchronous and asynchronous training depends on various factors, such as the specific distributed training framework being used, the hardware configuration, the size of the model and dataset, and the desired trade-offs between convergence guarantees and training speed. Synchronous training is generally simpler to implement and provides stronger convergence guarantees, while asynchronous training can offer faster training speed but may require additional measures to address parameter inconsistency.


## Here are some distribution strategies available in TensorFlow:

1. **MirroredStrategy:** This strategy is used for synchronous training on multiple GPUs within a single machine. It creates copies of the model on each GPU and divides the input data among them. The gradients are computed on each GPU independently and then aggregated to update the model weights. MirroredStrategy uses all-reduce algorithms to synchronize the variables and gradients across GPUs.

![multi-gpu-system](https://github.com/sandeep4055/Tensorflow/assets/70133134/665f8893-a7c0-4c1f-a1c2-9deaccd4b1f4)


2. **ParameterServerStrategy:** This strategy is designed for training on multiple machines. It involves multiple workers and one or more parameter servers. The model variables are split across parameter servers, and each worker is responsible for computing gradients on a subset of the training data. The gradients are then sent to the parameter servers for aggregation and updating the model weights.

![parameter-server-strategy](https://github.com/sandeep4055/Tensorflow/assets/70133134/5145a790-04da-4788-a612-54a167de59e7)


3. **MultiWorkerMirroredStrategy:** This strategy extends MirroredStrategy to training on multiple machines with multiple GPUs. It combines the synchronous training of MirroredStrategy with the distributed training of ParameterServerStrategy. It can be used in scenarios where you have multiple workers with multiple GPUs on each worker.

![system-cluster](https://github.com/sandeep4055/Tensorflow/assets/70133134/3382db45-afa3-4247-81ea-25d520c5e92a)


4. **CentralStorageStrategy:** This strategy is used for training on a single machine with multiple GPUs. It places variables on one device (typically the CPU) and keeps a single copy of the model on that device. Gradients are computed on each GPU independently and then aggregated on the CPU to update the model weights.

![central-storage-strategy](https://github.com/sandeep4055/Tensorflow/assets/70133134/eaf85a84-e7fd-448e-ac25-0a268465b2a1)


These distribution strategies help in scaling up training to larger models and datasets, reducing training time, and utilizing available hardware resources efficiently. They provide flexibility to choose the strategy based on the hardware configuration and the scale of your training setup.


## tf.distribute

tf.distribute is a module in TensorFlow that provides a high-level API for distributing training across multiple devices and machines. It offers various strategies for distributed training, including data parallelism, parameter server training, and custom strategies.

The tf.distribute module allows you to scale your training to multiple GPUs or machines, which can significantly improve training speed and model performance. It handles the distribution of data, computation, and model parameters across devices or machines, and provides mechanisms for synchronization and communication between them.

#### Key components and features of tf.distribute include:

1. **Strategies:** tf.distribute.Strategy is an abstract base class that defines the interface for distributing training. It provides different strategies such as MirroredStrategy, TPUStrategy, and MultiWorkerMirroredStrategy. These strategies handle the distribution of variables, gradients, and computation across multiple devices or machines.

2. **Data Distribution:** tf.distribute provides APIs to distribute and preprocess data efficiently. It offers methods for distributed input pipelines, including sharding, batching, and prefetching data across devices or machines.

3. **Model Training:** tf.distribute enables the distribution of the training process across devices or machines. It allows you to define and compile your models within the context of a strategy, automatically distributing the computation and managing the synchronization of gradients and updates.

4. **Custom Training Loops:** tf.distribute supports custom training loops, allowing you to have fine-grained control over the training process while leveraging distributed training capabilities. It provides APIs to run distributed computations, aggregate metrics, and synchronize state across devices or machines.

5. **Checkpointing and Saving:** tf.distribute handles saving and restoring models in a distributed setting. It ensures that model parameters and optimizer states are correctly saved and restored across devices or machines.

By using the functionalities provided by tf.distribute, you can effectively utilize multiple devices or machines to train your models, speeding up training and enabling larger models to be trained on larger datasets. It simplifies the process of distributed training and allows you to take advantage of the full power of your hardware infrastructure.


## what are functions available in tf.distrubute

The **tf.distribute** module in TensorFlow provides various functions and classes for distributed training. Here are some important functions available in tf.distribute:

- **tf.distribute.Strategy:** This is a class that represents the distribution strategy. It provides methods for distributing variables, gradients, and computations across multiple devices or machines. Some commonly used strategies include MirroredStrategy, TPUStrategy, and MultiWorkerMirroredStrategy.

1. **tf.distribute.experimental.TPUStrategy.initialize_tpu_system:** This function initializes the TPU system for distributed training on Google Cloud TPUs.

2. **tf.distribute.experimental.TPUStrategy.extended:** This function returns the extended form of the given strategy, which provides additional functionality specific to TPUs.

3. **tf.distribute.experimental.CentralStorageStrategy:** This is a strategy that supports training on multiple devices within a single machine. It allows sharing variables between devices and performs synchronization using a central storage.

4. **tf.distribute.experimental.MultiWorkerMirroredStrategy:** This is a strategy for training models on multiple workers in a synchronous data-parallel manner. It uses a parameter server approach to distribute variables and gradients across workers.

5. **tf.distribute.experimental.CollectiveCommunication.RING:** This function returns a CollectiveCommunication object representing the ring-based collective communication method used for communication between devices or machines.

6. **tf.distribute.cluster_resolver.TPUClusterResolver:** This class helps to resolve the TPU cluster information when training on TPUs. It can be used to create a TPU strategy and configure the training cluster.

7. **tf.distribute.OneDeviceStrategy:** This is a strategy that places all variables and computations on a single specified device. It is useful when you have only one device available or when you want to isolate computations on a specific device.

These are just a few examples of functions available in tf.distribute. The module provides many more functions and classes to support various distributed training scenarios, including data parallelism, parameter server training, and custom strategies.










