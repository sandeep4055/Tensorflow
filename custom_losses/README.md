# This Folder consists of code for creating Custom Loss Functions

Loss Functions : It's a method of evaluating how well your algorithm models your dataset.

Defining Loss function in the model :

# 1st way : model.compile(loss="mse", optimizer="sgd")

# 2nd way : model.compile(loss=tf.keras.losses.MSE, optimizer="sgd") 

### Note the 2nd way is flexible bcz we can pass parameters model.compile(loss=tf.keras.losses.MSE(param=value), optimizer="sgd") 