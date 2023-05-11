# Callbacks

- In TensorFlow, a callback is a set of functions that can be applied at various stages of training (e.g., at the beginning or end of an epoch, before or after a single batch, etc.) to perform certain operations. Callbacks are used to improve the performance of a model by modifying its behavior during training.

- There are several built-in callback functions in TensorFlow, such as ***ModelCheckpoint***, *** EarlyStopping*** , ***TensorBoard***, ***LearningRateScheduler***, etc.

- ModelCheckpoint is used to save the model weights after every epoch or only when the model has improved on a validation set. 
- EarlyStopping is used to stop the training process if the validation loss doesn't improve for a certain number of epochs. 
- TensorBoard is used to visualize the training metrics, such as the loss and accuracy, over time. 
- LearningRateScheduler is used to adjust the learning rate during training.

In addition to the built-in callbacks, users can also define their own custom callbacks by subclassing the tf.keras.callbacks.Callback class and implementing the desired functions. This allows for greater flexibility and customization in the training process.