
######### with scipy/numpy:
from scipy.stats import multivariate_normal
import numpy as np

x = np.array([[-1, -3], [0, 8], [-2, 10]])
mu = x.mean(axis=0)

multivariate_normal.pdf(x, mean=mu)
#array([  2.01556455e-15,   1.07237757e-03,   3.59742598e-07])

###################



######### with scipy/numpy:
from scipy.stats import multivariate_normal
import numpy as np

x = np.array([[-1, -3], [0, 8], [-2, 10]])
mu = x.mean(axis=0)
sigma = np.cov(x, rowvar=False)
multivariate_normal.pdf(x, mean=mu, cov=sigma)

#array([ 0.01179424,  0.01179424,  0.01179424])

###################



######### with tensorflow:
import tensorflow as tf
import numpy as np
tfd = tf.contrib.distributions
x = np.array([[-1, -3], [0, 8], [-2, 10]])
X = tf.constant(x, dtype=tf.float64, name="X")
mu = tf.constant(x.mean(axis=0), name="mu")

mvn = tfd.MultivariateNormalDiag(
    loc=mu)
with tf.Session() as sess:
  mvn.prob(X).eval()

#array([  2.01556455e-15,   1.07237757e-03,   3.59742598e-07])



### proper covariance
import tensorflow as tf
import numpy as np
tfd = tf.contrib.distributions

X = tf.constant(x, dtype=tf.float64, name="X")
mu = tf.constant(x.mean(axis=0), name="mu")
sigma = tf.constant(np.cov(x, rowvar=False))

mvn = tfd.MultivariateNormalFullCovariance(
    loc=mu,
    covariance_matrix=sigma)

with tf.Session() as sess:
  mvn.prob(X).eval()
#array([ 0.01179424,  0.01179424,  0.01179424])

##########################
