#!/usr/bin/python3

''' from tutorial:
https://www.tensorflow.org/tutorials/mnist/pros/
'''

import tensorflow as tf

BATCH_SIZE = 100
NUM_TRAINING_ITERATIONS = 100
HIDDEN_LAYER_SIZE = 25

def get_data():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  return mnist


def main():
  mnist = get_data()
  sess = tf.InteractiveSession()

  # placeholders for x and y
  x = tf.placeholder(tf.float32, shape=[None, 784])
  yhat = tf.placeholder(tf.float32, shape=[None, 10])

  # Variables for weights and bias
  W1 = tf.Variable(tf.random_normal([784, HIDDEN_LAYER_SIZE], stddev=0.1))
  W2 = tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE, 10], stddev=0.1))

  b1 = tf.Variable(tf.zeros([HIDDEN_LAYER_SIZE]))
  b2 = tf.Variable(tf.zeros([10]))

  # regression model
  z2 = tf.add(tf.matmul(x,W1), b1)
  a2 = tf.nn.sigmoid(z2)
  y = tf.add(tf.matmul(a2,W2), b2)

  # cost function -- cross-entropy between target 
  # and softmax activation function
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, yhat))

  # backprop via gradient descent with .5 steps
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

  # initialize variables for use within session
  sess.run(tf.global_variables_initializer())

  # run training iterations
  for i in range(NUM_TRAINING_ITERATIONS):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/BATCH_SIZE)
    for j in range(total_batch):
      batch = mnist.train.next_batch(BATCH_SIZE)
      # print('batch[0] shape: %s, batch[1] shape: %s' % (batch[0].shape, batch[1].shape))
      _, c = sess.run([optimizer, cost], feed_dict={ x: batch[0], yhat: batch[1] })
      avg_cost += c / total_batch
    print("Iteration: %d, cost: %.4f" % (i, avg_cost))
  # get boolean array of true/false for equality of predicted (y) equals actual (yhat)
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))

  # cast to float and get mean of correct predictions
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  print("Accuracy: %f" % accuracy.eval(feed_dict={x: mnist.test.images, yhat: mnist.test.labels}))


if __name__ == '__main__':
  main()
