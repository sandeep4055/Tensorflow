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






