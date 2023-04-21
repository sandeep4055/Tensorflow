import tensorflow as tf

tensor = tf.constant([
                      [1, 2, 3],
                      [3, 4, 5],
                      [6, 7, 8]
                      ])

print(tf.math.reduce_max(tensor,axis=0))

#print(tf.random.normal(mean=50,stddev=3,shape=(10,10)))