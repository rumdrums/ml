import numpy as np

def cost(X, y, theta):
	""" return cost, given matrices for X, y, theta """

	m = y.shape[0]
	predictions = X * theta
	squared_errors = np.square((predictions-y))
	numerator = 1.0/(2*m)
	cost = numerator * squared_errors.sum()
	return cost

