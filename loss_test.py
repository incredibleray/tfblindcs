import tensorflow as tf
import numpy as np

class LossTest(tf.test.TestCase):
  def test1(self):
    with self.session():
      m=tf.eye(2)
      self.assertAllClose(tf.norm(m), np.sqrt(2))

  def test2(self):
    with self.session():
      m=tf.eye(2)
      self.assertAllClose(tf.norm(m, axis=[-2,-1]), np.sqrt(2))

  def test3(self):
    with self.session():
      m=tf.eye(2)
      self.assertAllClose(tf.norm(m, axis=0), tf.constant([1, 1]))

  def test4(self):
    with self.session():
      A=tf.eye(2)
      B=tf.zeros([2,2])
      D=tf.abs(A)-tf.abs(B)
      self.assertAllClose(tf.norm(D), np.sqrt(2))

  def test5(self):
    with self.session():
      A=tf.eye(2)
      B=tf.eye(2)
      D=tf.abs(A)-tf.abs(B)
      self.assertAllClose(tf.norm(D), 0)
#   def testExponentiallyWeightedLoss(self):
#     with self.session():
#       label = tf.zeros([3, 1])
#       p2=tf.constant([[1.0], [0.0], [0.0]])
#       result = tf_lstm.decay_log1p_loss(p2, label).eval()
#       self.assertAlmostEqual(result, 0.49*np.log(2)/3)

#   def testExponentiallyWeightedLossStep(self):
#     with self.session():
#       label = tf.zeros([3, 1])
#       predictions=tf.constant([[0.0], [0.0], [1.0]])
#       weight=tf.math.cumprod(tf.fill(tf.shape(predictions), 0.7), reverse=True, exclusive=True)
#       self.assertAllClose(weight.eval(), tf.constant([[0.49], [0.7], [1.0]]))
      
#       log1p_loss=tf.log1p(tf.square(label-predictions))
#       self.assertAllClose(log1p_loss.eval(), tf.constant([[0.0], [0.0], [np.log(2)]]))

#       loss=tf.losses.compute_weighted_loss(log1p_loss,weight)
#       self.assertAlmostEqual(loss.eval(), np.log(2)/3)

if __name__ == '__main__':
  tf.test.main()