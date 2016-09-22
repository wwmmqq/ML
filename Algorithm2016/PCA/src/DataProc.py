# -*- coding: utf-8 -*-
import numpy as np

class DataProc(object):
	def __init__(self, filename = 'iris'):
		path = '../Data/'+ filename + '.data.txt'
		lines = map(lambda x : x.split('\n')[0], open(path).readlines())
		pre_data = map(lambda line : line.split(',')[:4], lines)

		self.x = np.transpose(np.mat(np.asarray(pre_data, dtype = np.float64)))
		
		y = []
		for i in range(50):
			y.append(0)
		for i in range(50):
			y.append(1)
		for i in range(50):
			y.append(2)
		self.y = np.mat(y)
