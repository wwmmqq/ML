from numpy import *
import random

def sigmoid( x ):
	return 1.0 / (1 + exp(-x))

def loadData(filestr):
	myData = []
	myLabel = []
	fr = open(filestr)
	for line in fr:
		lineArr = line.strip().split()
		tmp = [float(x) for x in lineArr]
		tmp.insert(0,1)
		myLabel.append(int(tmp.pop(-1)))
		myData.append(tmp)

	return mat(myData), mat(myLabel).transpose()


def gradAscent(train_x, train_y, maxCycles=100):

	alpha = 0.01
	maxCycles = 30
	#m, n = shape(train_x)
	numSamples, numFeatures = shape(train_x)
	w = ones((numFeatures, 1))

	for k in range(maxCycles):
		h = sigmoid(train_x * w)
		error = h - train_y # also train_y - sigmoid(h) is ok
		w = w + alpha * train_x.transpose()*error
	return w


def stocGradAscent0(train_x, train_y, maxCycles=100):
	alpha = 0.01
	numSamples, numFeatures = shape(train_x)
	w  = ones((numFeatures, 1))
	for i in range(numSamples):
		h = sigmoid(sum(train_x[i, :] * w))#sum remove is ok, just make np(1,1) to a number
		error = train_y[i, 0] - h
		w = w + alpha * train_x[i, :].transpose() * error
	return w


def stocGradAscent1(train_x, train_y, maxCycles=100):
	numSamples, numFeatures = shape(train_x)
	w = ones((numFeatures, 1))
	dataIndex = range(numSamples)

	for k in range(maxCycles):
	    for i in range(numSamples):
	        alpha = 4.0 / (1.0 + k + i) + 0.01  
	        randIndex = int(random.uniform(0, len(dataIndex)))
	        h = sigmoid(sum(train_x[randIndex, :] * w)) 
	        error = train_y[randIndex, 0] - h
	        w = w + alpha * train_x[randIndex, :].transpose() * error  
	        del(dataIndex[randIndex]) # during one interation, delete the optimized sample
	    dataIndex = range(numSamples)
	return w


# test your trained Logistic Regression model given test set  
def predict(weights, test_x, test_y):
    numSamples, numFeatures = shape(test_x)  
    matchCount = 0

    for i in xrange(numSamples):
        p = sigmoid(test_x[i, :] * weights) > 0.5  #[0, 0]
        if p == bool(test_y[i, 0]):
            matchCount += 1

    accuracy = float(matchCount) / numSamples  
    return accuracy


def main():
	myData, myLabel = loadData('train.txt')
	#weights = gradAscent(myData, myLabel)
	weights = stocGradAscent0(myData, myLabel)

	testData, testLabel = loadData('test.txt')
	accuracy = predict(weights, testData, testLabel)
	print 'The classify accuracy is: %.3f' % (accuracy) 

if __name__ == '__main__':
	main()

