import numpy as np
import cost_function

def gradient_descent(X, y, theta, alpha, iters):
	""" do gradient descent on given datasets """

	m = y.shape[0]
	cost_array = np.matrix(np.zeros([iters,1])) 	
	for i in range(iters):
		delta = X.transpose() * (X * theta - y)
		# update theta based on alpha:
		theta = theta - (alpha/m * delta)
		cost_array[i] = cost_function.cost(X, y, theta) 
		print("Iteration %d: Cost is %.4f, slope is %.4f" % (i, cost_array[i], theta[1]))	
