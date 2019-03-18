from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

y=tf.constant(np.random.rand(3, 1), dtype=tf.float32)
X=tf.constant(np.random.rand(3, 2), dtype=tf.float32)

D=tf.get_variable('D',[2, 2])
z=tf.get_variable('z',[2, 1])

XD=tf.matmul(X,D)

residual=y-tf.matmul(XD, z)

lambda_l1=0.0000001
# linreg_loss=tf.norm(residual)
# loss =linreg_loss+lambda_l1*tf.norm(z, ord=1)
loss=tf.norm(residual)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.

opt=tf.train.ProximalGradientDescentOptimizer(learning_rate, l1_regularization_strength=lambda_l1)
# The Gradient Descent Optimizer does the heavy lifting
D_train_op = opt.minimize(loss, global_step=global_step, var_list=[D,])
z_train_op = opt.minimize(loss, global_step=global_step, var_list=[z,])
# linreg_learning_rate=0.1
# z_train_op = tf.train.FtrlOptimizer(linreg_learning_rate, l1_regularization_strength=lambda_l1).minimize(loss, var_list=[z,])

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(model)
  # A=sess.run(A)
  # np.savetxt('A_init.csv', A, delimiter=',')

  for i in range(5000):
    sess.run(z_train_op)

    if i%1000==0:
      print('Step #' + str(i) + ' Loss = ' + str(sess.run(loss)))
      sess.run(D_train_op)
      print('Step #' + str(i+1) + ' Loss = ' + str(sess.run(loss)))
    
    this_loss = sess.run(loss)

    if this_loss < 1e-9:
      break

  X, y, D, z=sess.run([X, y, D, z])
  np.savetxt('X.csv', X, delimiter=',')
  np.savetxt('y.csv', y, delimiter=',')
  np.savetxt('D.csv', D, delimiter=',')
  np.savetxt('z.csv', z, delimiter=',')
