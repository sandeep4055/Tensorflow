# This folder consists of code about how to implement Custom layers in tensorflow

## What are Layers in Tensorflow?
In TensorFlow, a layer is a fundamental building block used to create neural networks. Layers are used to define the input and output shapes of the data and to apply operations on this data. There are various types of layers available in TensorFlow, such as:

1. Input layer: This layer defines the shape of the input data. It is the first layer of the network.

2. Dense layer: This layer is also known as a fully connected layer. It applies a matrix multiplication operation on the input data.

3. Convolutional layer: This layer applies a convolution operation on the input data.

4. Pooling layer: This layer is used to reduce the spatial dimensions of the input data by applying pooling operations such as max pooling or average pooling.

5. Recurrent layer: This layer is used for processing sequential data such as time series or natural language processing.

6. Dropout layer: This layer randomly drops out a fraction of the input units during training to prevent overfitting.

7. Batch normalization layer: This layer normalizes the input data by adjusting and scaling the activations.

Layers can be stacked on top of each other to create complex neural network architectures.

## Inbuilt vs Custom layers
In TensorFlow, there are two types of layers: inbuilt layers and custom layers.

**Inbuilt layers** are predefined layers that come with TensorFlow, such as Dense, Conv2D, MaxPooling2D, Dropout, etc. These layers are easy to use and provide good performance out of the box. They are designed to handle most of the common use cases in deep learning.

**Custom layers**, on the other hand, are user-defined layers that allow you to implement your own neural network architectures. They are useful when you need to create a layer that is not available in TensorFlow's inbuilt layers or when you want to customize an existing layer. Custom layers can be implemented using TensorFlow's low-level API, such as tf.keras.layers.Layer or tf.keras.layers.Lambda, and they offer flexibility in designing and implementing neural network architectures.
