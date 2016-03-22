#!/usr/bin/python

import numpy as np
import cost_function as cf
import sys

the_file='ex1data1.txt'

def main():
	try:
		data = np.loadtxt(open(the_file, 'r'), delimiter=',')
	except:
		print('Failed to load data')
		sys.exit(1)

	X = data[:,0]
	y = data[:,1]

	theta = np.matrix('0;0')
	cost = cf.cost(X,y,theta)
	print('cost = %d' % cost)		

main()
