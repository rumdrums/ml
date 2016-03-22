import numpy as np

def cost(X, y, theta):
	""" return cost, given matrices for X, y, theta """

	m = y.shape[0]
	predictions = X * theta
	squared_errors = np.square((predictions-y))
	cost = 1/(2*m) * squared_errors
	return cost

