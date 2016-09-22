# -*- coding: utf-8 -*-

import numpy as np

def PCA(X, k):
	# X : row for features,  low for samples
	zero_mean_X = X - np.mean(X, axis=1)
	#or zero_mean_X =np.mat(map(lambda line : map(lambda x: x - np.mean(line, axis=0), line), X))
	C = np.cov(zero_mean_X, rowvar=1) #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
	W, V = np.linalg.eig(np.mat(C))	#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量 
	print 'k = {} W: '.format(k)
	print W
	print W.shape
	topk_index = np.argsort(W)[-1:-k-1:-1]	#对特征值从小到大排序
	P = np.transpose(V[:,topk_index])
	Y = P * zero_mean_X
	return Y