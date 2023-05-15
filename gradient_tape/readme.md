# Gradient Tape
In TensorFlow, differentiation is commonly performed using the tf.GradientTape API, which allows you to compute gradients of functions with respect to their input variables. This is especially useful in deep learning for calculating gradients during backpropagation.

##### The tf.GradientTape API in TensorFlow provides several functions that can be used within the tape context to compute gradients. Here are some of the commonly used functions available in tf.GradientTape:

1. **tape.watch(variable):** This function explicitly tells the tape to watch a specific variable, even if it's not a trainable variable. It is useful when you want to compute gradients with respect to non-trainable variables.

2.  **tape.gradient(target, sources):** This function computes the gradients of the target tensor with respect to the sources. The target can be a single tensor or a list of tensors, and the sources can be a single tensor or a list of tensors.

3.  **tape.gradient(target, sources, output_gradients):** This function computes the gradients of the target tensor with respect to the sources, while also considering the provided output gradients. The output_gradients can be a single tensor or a list of tensors.

4. **tape.jacobian(target, sources, parallel_iterations=None, experimental_use_pfor=True):** This function computes the Jacobian matrix of the target tensor with respect to the sources. The target and sources can be single tensors or lists of tensors. The parallel_iterations and experimental_use_pfor arguments are optional and control the parallelization of the computation.

5. **tape.batch_jacobian(target, sources, parallel_iterations=None, experimental_use_pfor=True):** This function computes the batched Jacobian matrix of the target tensor with respect to the sources. The target and sources can be single tensors or lists of tensors. The parallel_iterations and experimental_use_pfor arguments are optional and control the parallelization of the computation.

6. **tape.gradient(target, sources, unconnected_gradients=tf.UnconnectedGradients.NONE):** This function computes the gradients of the target tensor with respect to the sources. The unconnected_gradients parameter determines the behavior when the target tensor doesn't depend on the sources.

7. **tape.batch_gradient(target, sources, output_gradients=None):** This function computes the gradients of the target tensor with respect to the sources in a batched manner. The target and sources can be single tensors or lists of tensors. The output_gradients can be a single tensor or a list of tensors.

8. **tape.reset():** This function resets the gradient tape, clearing any previously recorded operations. It is useful when you want to start a fresh gradient computation.

9. **tape.stop_recording():** This function stops recording any new operations on the gradient tape. It can be used to temporarily pause the tape recording and reduce memory consumption.

10. **tape.pause():** This function pauses the tape recording, similar to tape.stop_recording(). However, it returns a tf.GradientTape object that can be used to resume the recording later.

