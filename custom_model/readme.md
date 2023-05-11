# Custom Models
- In TensorFlow, custom models can be created by subclassing the tf.keras.Model class. This provides a way to create models with custom architecture that can be trained and evaluated using the same API as built-in Keras models.

- To create a custom model, we need to define the layers in the __init__ method and define the forward pass (also called the call method) in the call method. The call method takes the input tensor as argument and applies the layers in sequence to produce the output tensor.

## Subclassing vs Functional Api

- Subclassing provides more flexibility than the Functional API in TensorFlow because it allows you to define your own forward pass rather than using predefined layers. This can be particularly useful when designing novel architectures or when the layers in your model do not follow a linear structure.

- Additionally, using subclassing allows you to implement more complex architectures that involve conditionals or loops, which are not possible with the Functional API.

- Another advantage of using subclassing is that it allows for more fine-grained control over the model's behavior, such as custom initialization of weights, custom losses, or custom metrics.

Overall, subclassing is a powerful tool that provides more control and flexibility when designing models in TensorFlow.