# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from DataProc import *
from PCA import *

data_set = DataProc('iris')


def draw_2d():
	x2 = PCA(data_set.x, 2)

	plt.figure()
	plt.scatter(x2[0,:50], x2[1,:50], marker = 'x', color = 'm', s = 30, label='Iris-setosa')
	plt.scatter(x2[0,50:100], x2[1,50:100], marker = '+', color = 'c', s = 50, label='Iris-versicolor')
	plt.scatter(x2[0,100:150], x2[1,100:150], marker = 'o', color = 'r', s = 15, label='Iris-virginica')
	plt.legend()
	plt.title('PCA of IRIS k = 2')
	plt.show()

def draw_3d():
	x3 = PCA(data_set.x, 3)
	ax = plt.subplot(111, projection='3d')
	for i in range(50):
		ax.scatter(x3[0,i], x3[1,i], x3[2,i], marker = 'x', c='m', s = 30)
	
	for i in range(50, 100):
		ax.scatter(x3[0,i], x3[1,i], x3[2,i], marker = '+', c='c', s = 30)

	for i in range(100, 150):
		ax.scatter(x3[0,i], x3[1,i], x3[2,i], marker = 'o', c='r', s = 30)

	#ax.scatter(x3[0, 50:100], x3[1, 50:100], x3[2, 50:100], marker = '+', c='c')
	#ax.scatter(x3[0, 100:150], x3[1, 100:150], x3[2, 100:150], marker = 'o', c='r')
	ax.set_zlabel('Z')
	ax.set_ylabel('Y')
	ax.set_xlabel('X')
	plt.title('PCA of IRIS k = 3')
	plt.show()

if __name__ == '__main__':
	#draw_2d()
	draw_3d()