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

# keras vs tf.keras

Keras and tf.keras are both high-level APIs for building and training deep learning models in TensorFlow. However, there are some differences between the two:

1. **Development and Maintenance:** Keras was originally an independent open-source project developed by François Chollet. In 2017, Keras was integrated into TensorFlow as its official high-level API, called tf.keras. tf.keras is now the recommended API for TensorFlow and is actively maintained by the TensorFlow team. The standalone Keras library is still available and maintained, but it may not have the latest updates and improvements from the TensorFlow ecosystem.

2. **Integration with TensorFlow:** tf.keras is tightly integrated with the TensorFlow ecosystem. It seamlessly works with other TensorFlow components, such as tf.data for data input pipelines, tf.distribute for distributed training, and tf.saved_model for model saving and deployment. It also supports TensorFlow-specific features, such as eager execution and TensorFlow 2.0's functions and operators.

3. **Compatibility:** tf.keras strives to maintain compatibility with the Keras API, but there might be some differences and additions in the TensorFlow implementation. Some TensorFlow-specific features, such as custom TensorFlow ops or distributed training strategies, may not be available in the standalone Keras library.

4. **Community and Support:** tf.keras benefits from the larger TensorFlow community, which provides extensive documentation, tutorials, and examples specific to TensorFlow. It also leverages TensorFlow's resources, such as TensorFlow Hub for pre-trained models and TensorFlow Addons for additional functionalities.

In summary, tf.keras is the recommended high-level API for TensorFlow. It offers better integration with TensorFlow and takes advantage of TensorFlow-specific features and optimizations. If you are working with TensorFlow, it is recommended to use tf.keras for building and training your models. However, if you have an existing codebase or project built with the standalone Keras library, it can still be used and maintained separately.

# Before the integration of Keras into TensorFlow

- Before the integration of Keras into TensorFlow, TensorFlow had its own low-level API for building and training deep learning models. This API provided more flexibility and control over the model architecture and training process but required more code and manual implementation of various components.

- The pre-Keras TensorFlow API consisted of lower-level classes and functions for defining the model architecture, specifying the loss function, optimizing the model parameters, and executing the training loops. It required developers to explicitly define and manage TensorFlow variables, placeholders, sessions, and gradients.

- While powerful and flexible, this low-level TensorFlow API had a steeper learning curve and required more boilerplate code to accomplish common deep learning tasks. It was not as user-friendly and intuitive as higher-level APIs like Keras.

- To address these concerns, TensorFlow decided to integrate Keras into its ecosystem. Keras is a high-level deep learning API that provides a user-friendly and intuitive interface for building and training models. It simplifies many aspects of model development and abstracts away the low-level TensorFlow details, making it easier and more accessible to developers.

- The integration of Keras into TensorFlow as tf.keras brought the best of both worlds. It retained the simplicity and ease of use of Keras while leveraging the power and ecosystem of TensorFlow. Developers could benefit from the high-level Keras API for most use cases while still having access to TensorFlow's lower-level features and optimizations when needed.

- Overall, the integration of Keras into TensorFlow expanded the TensorFlow ecosystem, making it more user-friendly and accessible to a wider range of developers, from beginners to experts. It provided a unified and consistent interface for building and training deep learning models within the TensorFlow framework.


#

| Number | Topic | link | 
| ----- |  ----- |  ----- |
| 1 |  History and overview of TensorFlow |  https://github.com/sandeep4055/Tensorflow/tree/main/history_tensorflow |
| 2 |  Tensors |  https://github.com/sandeep4055/Tensorflow/tree/main/basics |
| 3 |  Model |  https://github.com/sandeep4055/Tensorflow/tree/main/model |
| 4 |  custom model |  https://github.com/sandeep4055/Tensorflow/tree/main/custom_model |
| 5 |  custom losses |  https://github.com/sandeep4055/Tensorflow/tree/main/custom_losses |
| 6 |  custom layers |  https://github.com/sandeep4055/Tensorflow/tree/main/custom_layer |
| 7 |  callbacks |  https://github.com/sandeep4055/Tensorflow/tree/main/callbacks |
| 8 |  Classification |  https://github.com/sandeep4055/Tensorflow/tree/main/classification |
| 9 |  Functional Api |  https://github.com/sandeep4055/Tensorflow/tree/main/functional_api |
| 10 |  Data Api  |  https://github.com/sandeep4055/Tensorflow/tree/main/data_api |
| 11 |  Gradient Tape  |  https://github.com/sandeep4055/Tensorflow/tree/main/gradient_tape |


