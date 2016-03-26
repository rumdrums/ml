#!/usr/bin/python

import numpy as np
import cost_function 
import gradient_descent
import sys

the_file='ex1data1.txt'

def main():
	try:
		data = np.loadtxt(open(the_file, 'r'), delimiter=',')
	except:
		print('Failed to load data')
		sys.exit(1)

	X = np.matrix(data[:,0]).transpose()
	y = np.matrix(data[:,1]).transpose()
	# add column of ones:
	X = np.insert(X, 0, values=1,axis=1)

	theta = np.matrix('0;0')
	alpha = .01
	iters = 1500
	gradient_descent.gradient_descent(X, y, theta, alpha, iters)

main()
