import tensorflow as tf

# Creating Custom Layers
# Lets create our custom dense layer

"""So, in short, the variables defined in the __init__ method are temporary and are not part of the layer's state,
 while the variables defined in the build method are part of the layer's state and need to be saved and restored.
"""

class My_Dense_Layer(tf.keras.layers.Layer):

    # Constructor
    def __init__(self, units=32, activations=None):
        super(My_Dense_Layer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activations)


    def build(self, input_shape):
        # initialize w & b
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel", initial_value=w_init(shape=(input_shape[-1], self.units)),
                             dtype=tf.float32,
                             trainable=True)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias", initial_value=b_init(shape=(self.units,)), dtype=tf.float32, trainable=True)

    def call(self, inputs):

        return self.activation(tf.matmul(inputs, self.w) + self.b)


layer = My_Dense_Layer(units=1, activations="tanh")

x = tf.ones((1, 1), dtype=tf.float32)

y = layer(x)

print(y)




