import tensorflow as tf

#unknown basis under which x is sparse
sparseBasis = tf.get_variable("U", [10, 10])

# sensing matrix
sensingMatrix = tf.constant(-1.0, shape=[2, 3])

