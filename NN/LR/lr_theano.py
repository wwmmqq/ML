import numpy
import theano
import theano.tensor as T
import six.moves.cPickle as pickle

rng = numpy.random

def load_data(file_path):
	
	with open(file_path) as f:
		x = []
		y = []
		for line in f:
			tmp = [numpy.float64(t) for t in line.split()]
			y.append(tmp.pop(-1))
			x.append(tmp)
		x =  numpy.array(x)
		y =  numpy.array(y)

	return (x, y)


def LR():
	N = 800 	# training sample size
	feats = 10	# number of input variables
	training_steps = 1000 # 1000

	D = load_data('../data/lr/train.txt')

	# Declare Theano symbolic variables
	x = T.dmatrix("x")
	y = T.dvector("y")

	# initialize the weight vector w randomly
	#
	# this and the following bias variable b
	# are shared so they keep their values
	# between training iterations (updates)
	w = theano.shared(rng.randn(feats), name="w")

	# initialize the bias term
	b = theano.shared(0., name="b")

	#print("Initial model:")
	#print(w.get_value())
	#print(b.get_value())

	# Construct Theano expression graph
	p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
	prediction = p_1 > 0.5                    # The prediction thresholded
	xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
	cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
	#cost = xent.sum() + 0.01 * (w ** 2).sum()# The cost to minimize
	gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
	                                          # w.r.t weight vector w and
	                                          # bias term b
	                                          # (we shall return to this in a
	                                          # following section of this tutorial)

	# Compile
	train = theano.function(
	          inputs=[x,y],
	          outputs=[prediction, xent],
	          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

	predict = theano.function(inputs=[x], outputs=prediction)


	# Train
	for i in range(training_steps):
	    pred, err = train(D[0], D[1])

	#print("Final model:")
	#print(w.get_value())
	#print(b.get_value())

	#test
	# save the best model
	with open('best_model.pkl', 'wb') as f:
		pickle.dump(classifier, f)

	#classifier = pickle.load(open('best_model.pkl'))
	testD = load_data('../data/lr/test.txt')
	pred = predict(testD[0])
	error_cnt = 0
	for i in range(len(pred)):
		if pred[i] != testD[1][i]:
			error_cnt += 1

	print error_cnt*1.0 / len(pred)
		

if __name__ == '__main__':
	LR()

