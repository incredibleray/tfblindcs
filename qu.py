from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

y=tf.constant(np.random.rand(3, 1))
X=tf.constant(np.random.rand(3, 2))

D=tf.get_variable('D',[2, 2])
z=tf.get_variable('z',[2, 1])

XD=tf.matmul(X,D)

residual=y-tf.matmul(XD, z)

lambda_l1=0.2
linreg_loss=tf.norm(residual)
loss =linreg_loss+lambda_l1*tf.norm(z, ord=1)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.

# The Gradient Descent Optimizer does the heavy lifting
D_train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=[D,])
# train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

linreg_learning_rate=0.02
z_train_op = tf.train.FtrlOptimizer(linreg_learning_rate, l1_regularization_strength=0.2).minimize(linreg_loss, var_list=[z,])

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
