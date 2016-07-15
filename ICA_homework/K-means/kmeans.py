# -*-coding: utf-8 -*-

import numpy as np
import random

def read_data(filepath='irisNoLabel.data'):
	date_set = []
	with open(filepath, 'r') as f:
		for eachLine in f:
			temp = [float(x) for x in eachLine.split(',')]
			date_set.append(temp)
	return np.array(date_set)

# v1 and v2 is np array
def distance(v1, v2):
	return np.sqrt(((v1 - v2)**2).sum())

#DATE is a numpy array
def init_centers(DATE, k):
	centers = []
	id = random.sample([x for x in range(len(DATE))], k)
	for x in id:
		centers.append(DATE[x])
	return np.array(centers)


def update_centers(clusters):
	newcenters = []
	for i in range(clusters.shape[0]):
		a_cluster = np.array(clusters[i])
		a_new_center = []
		for x in range(a_cluster.shape[1]):
			a_new_center.append( sum(a_cluster[:,x])/a_cluster.shape[0])
		newcenters.append(a_new_center)

	return np.array(newcenters)

def update_clusters(DATE, centers):
	clusters = [[] for i in range(centers.shape[0])]

	for i in range(DATE.shape[0]):
		shortest = float("inf")
		c = None
		for j in range(centers.shape[0]):
			dist = distance(DATE[i], centers[j])
			if dist < shortest:
				shortest = dist
				c = j
		clusters[c].append(DATE[i])
	return np.array(clusters)

def kmeans(DATE, k=3, batchsize=100):
	centers = init_centers(DATE, k)
	
	clusters = None
	for batch in xrange(batchsize):
		clusters = update_clusters(DATE, centers)
		centers = update_centers(clusters)
	return np.array(clusters)


def accurate(DATE, result):

	flag = [[], [], []]
 	for i in range(result.shape[0]):
		for j in range(len(result[i])):
			for k in range(DATE.shape[0]):
				if((DATE[k] == result[i][j])[1]):
					if(k >= 0 and k <= 49 ):
						flag[i].append(1)
					elif(k >= 50 and k <= 99 ):
						flag[i].append(2)
					if(k >= 100 and k <= 149 ):
						flag[i].append(3)
					break
	
	return  sum([max([y.count(x) for x in (1,2,3)]) for y in flag])/ 150.0


def main():
	import time
	t1 = time.time()
	DATE = read_data('irisNoLabel.data')
	t2 = time.time()
	result =  kmeans(DATE, 3, 100)
	t3 = time.time()

	print "clusters1 size : " + str(len(result[0]))
	print "clusters2 size : " + str(len(result[1]))
	print "clusters3 size : " + str(len(result[2]))
	print "accurate : " + str(accurate(DATE, result))
	print "read date cost time: " + str(t2 -t1)
	print "kmeans run time: " + str(t3 - t2)
if __name__ == '__main__':
	main()