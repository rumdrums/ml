#!/usr/bin/python3

''' from tutorial:
https://www.tensorflow.org/tutorials/mnist/pros/
'''

import tensorflow as tf

tf.set_random_seed(0)

BATCH_SIZE = 100
NUM_TRAINING_ITERATIONS = 10000

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
  W1 = tf.Variable(tf.truncated_normal([784,200], stddev=0.1))
  W2 = tf.Variable(tf.truncated_normal([200,100], stddev=0.1))
  W3 = tf.Variable(tf.truncated_normal([100,60], stddev=0.1))
  W4 = tf.Variable(tf.truncated_normal([60,30], stddev=0.1))
  W5 = tf.Variable(tf.truncated_normal([30,10], stddev=0.1))

  B1 = tf.Variable(tf.zeros([200]))
  B2 = tf.Variable(tf.zeros([100]))
  B3 = tf.Variable(tf.zeros([60]))
  B4 = tf.Variable(tf.zeros([30]))
  B5 = tf.Variable(tf.zeros([10]))

  # regression model
  y1 = tf.nn.sigmoid(tf.matmul(x,W1) + B1)
  y2 = tf.nn.sigmoid(tf.matmul(y1,W2) + B2)
  y3 = tf.nn.sigmoid(tf.matmul(y2,W3) + B3)
  y4 = tf.nn.sigmoid(tf.matmul(y3,W4) + B4)
  y_logits = tf.matmul(y4,W5) + B5
  y = tf.nn.softmax(y_logits)

  # cost function -- cross-entropy between target
  # and softmax activation function
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_))*100.0

  # training via gradient descent
  train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)

  # initialize variables for use within session
  sess.run(tf.global_variables_initializer())

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

