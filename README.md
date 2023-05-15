# Tensorflow
<p>TensorFlow is an open-source framework from Google for creating Machine Learning models. Although the software is written in C++, it is otherwise language-independent and can therefore be used very easily in various programming languages. For many users, the library has now become the standard for Machine Learning, since common models can be built comparatively simply. In addition, state-of-the-art ML models can also be used via TF, such as various transformers.</p>

<img src="https://user-images.githubusercontent.com/70133134/233819149-5246e572-7792-4be5-846b-de4ed8854868.gif" height="450" width="800%">

# What are Tensors?
<p>The name TensorFlow may seem a bit strange at first since there is no direct connection to Machine Learning. However, the name comes from the so-called tensors, which are used to train Deep Learning models and therefore form the core of TF.</p>

<p>The tensor is a mathematical function from linear algebra that maps a selection of vectors to a numerical value. The concept originated in physics and was subsequently used in mathematics. Probably the most prominent example that uses the concept of tensors is general relativity.</p>

<img src="https://user-images.githubusercontent.com/70133134/233819282-d6dec057-7260-4867-95a0-1c0047351070.png" height="450" width="80%">

<p>In the field of Machine Learning, tensors are used as representations for many applications, such as images or videos. In this way, a lot of information, some of it multidimensional, can be represented in one object. An image, for example, consists of a large number of individual pixels whose color value in turn is composed of the superposition of three color layers (at least in the case of RGB images). This complex construction can be represented compactly with a tensor.</p>

#### [How to create tensors? click me...](https://github.com/sandeep4055/Tensorflow/tree/main/basics#readme)

# High level Api's vs Low level Api's in Tensorflow

In TensorFlow, high-level and low-level APIs refer to different levels of abstraction and functionality provided by the framework.

- **High-level APIs:** High-level APIs provide a more user-friendly and simplified interface for building and training machine learning models. These APIs are designed to abstract away the low-level details and provide pre-built components and functions that make it easier to work with TensorFlow. Some popular high-level APIs in TensorFlow include:

1. **Keras:** Keras is a high-level neural networks API that provides a user-friendly interface for building, training, and deploying deep learning models. It offers a wide range of pre-built layers, optimizers, and loss functions, making it easy to define and train models.

2. **tf.keras:** tf.keras is TensorFlow's implementation of the Keras API. It integrates seamlessly with other TensorFlow functionalities and provides additional features and capabilities specific to TensorFlow.

3. **Estimators:** TensorFlow Estimators are a high-level API for building TensorFlow models. They provide a simplified interface for training, evaluating, and deploying models, and they encapsulate much of the boilerplate code required for common tasks.

High-level APIs are often preferred for their simplicity, ease of use, and productivity, especially for beginners or when prototyping models.

- **Low-level APIs:** Low-level APIs provide more flexibility and control over the model-building process. They offer lower-level operations and abstractions that allow users to define custom models, loss functions, and training procedures. The low-level APIs in TensorFlow include:

1. **TensorFlow Core:** TensorFlow Core is the foundational library of TensorFlow that provides the basic building blocks for constructing machine learning models. It offers operations for mathematical computations, tensor manipulation, and model optimization.

2.**tf.GradientTape:**tf.GradientTape is a low-level API for automatic differentiation, which is essential for computing gradients during training. It enables users to define custom training loops and compute gradients of variables with respect to a given loss function.

3. **tf.data:** tf.data is a low-level API for building efficient input pipelines for training models. It provides functionality for reading and preprocessing data, batching, shuffling, and iterating over datasets.

Low-level APIs are typically used when you need fine-grained control over the model architecture, training process, or when implementing custom algorithms or layers.

Both high-level and low-level APIs have their own advantages and use cases. High-level APIs are generally easier to use and provide faster development and prototyping, while low-level APIs offer more flexibility and customization options. The choice between them depends on the specific requirements and complexity of your machine learning project.



| Number | Topic | link | 
| ----- |  ----- |  ----- |
| 1 |  Tensors |  https://github.com/sandeep4055/Tensorflow/tree/main/basics |
| 2 |  Model |  https://github.com/sandeep4055/Tensorflow/tree/main/model |
| 3 |  custom model |  https://github.com/sandeep4055/Tensorflow/tree/main/custom_model |
| 4 |  custom losses |  https://github.com/sandeep4055/Tensorflow/tree/main/custom_losses |
| 5 |  custom layers |  https://github.com/sandeep4055/Tensorflow/tree/main/custom_layer |
| 6 |  callbacks |  https://github.com/sandeep4055/Tensorflow/tree/main/callbacks |
| 7 |  Classification |  https://github.com/sandeep4055/Tensorflow/tree/main/classification |
| 8 |  Functional Api |  https://github.com/sandeep4055/Tensorflow/tree/main/functional_api |
| 9 |  Data Api  |  https://github.com/sandeep4055/Tensorflow/tree/main/data_api |
| 10 |  Gradient Tape  |  https://github.com/sandeep4055/Tensorflow/tree/main/gradient_tape |



