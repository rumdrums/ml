#!/usr/bin/python3

''' from tutorial:
https://www.tensorflow.org/tutorials/mnist/pros/
'''

import tensorflow as tf

BATCH_SIZE = 100
NUM_TRAINING_ITERATIONS = 1000

def get_data():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  return mnist


def main():
  mnist = get_data()
  sess = tf.InteractiveSession()

  # placeholders for x and y
  x = tf.placeholder(tf.float32, shape=[None, 784])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])

  # Variables for weights and bias
  W = tf.Variable(tf.zeros([784,10]))
  b = tf.Variable(tf.zeros([10]))

  # initialize variables for use within session
  sess.run(tf.global_variables_initializer())

  # regression model
  y = tf.matmul(x,W) + b

  # cost function -- cross-entropy between target
  # and softmax activation function
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

  # training via gradient descent with .5 steps
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # run training iterations
  for i in range(NUM_TRAINING_ITERATIONS):
  	batch = mnist.train.next_batch(BATCH_SIZE)
  	# print('batch[0] shape: %s, batch[1] shape: %s' % (batch[0].shape, batch[1].shape))
  	train_step.run(feed_dict={ x: batch[0], y_: batch[1] })


  # get boolean array of true/false for equality of predicted (y) equals actual (y_)
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

  # cast to float and get mean of correct predictions
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # accuracy should be about 92 %
  print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
  main()

