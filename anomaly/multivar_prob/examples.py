
######### with scipy/numpy:
from scipy.stats import multivariate_normal
import numpy as np

x = np.array([[-1, -1], [0, 0], [-2, -2]])
mu = [-1, -1]
#sigma = [1, 2.]

multivariate_normal.pdf(x, mean=mu)
#array([ 0.15915494,  0.05854983,  0.05854983])

###################




######### with tensorflow:
import tensorflow as tf
import numpy as np
tfd = tf.contrib.distributions
X = tf.constant(np.array([[-1., -1.], [0., 0.], [-2., -2.]]), name="X")
mu = tf.constant(np.array([-1., -1.]), name="mu")

mvn = tfd.MultivariateNormalDiag(
    loc=mu)
with tf.Session() as sess:
  mvn.prob(X).eval()

#array([ 0.15915494,  0.05854983,  0.05854983])

##########################

