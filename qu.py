from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

m=np.loadtxt('A_init.csv', delimiter=',')
# A_init=tf.Variable(m, name='A')

A=tf.get_variable('A', initializer=m)
q, r = tf.qr(A)
d, u, v = tf.svd(A)

diff=tf.abs(q)-tf.abs(u)
loss =tf.norm(diff)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.

# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(model)
  # A=sess.run(A)
  # np.savetxt('A_init.csv', A, delimiter=',')

  for i in range(50000):
    sess.run(train_op)
    this_loss = sess.run(loss)

    if (i+1)%1000==0:
      print('Step #' + str(i+1) + ' Loss = ' + str(this_loss))

    if this_loss < 1e-9:
      break

  A, q, u=sess.run([A, q, u])
  np.savetxt('A.csv', A, delimiter=',')
  np.savetxt('q.csv', q, delimiter=',')
  np.savetxt('u.csv', u, delimiter=',')
