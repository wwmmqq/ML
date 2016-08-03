import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T

import mnist

def load_data(d_type='train'):
    mn = mnist.MNIST('../data/mnist')
    #[(train_set_x, train_set_y), (test_set_x, test_set_y)]
    if d_type == 'train':
        return mn.load_training()
    elif d_type == 'test':
        return mn.load_testing()

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(name='W', value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(name='b', value=numpy.zeros( (n_out,), dtype=theano.config.floatX), borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        #axis - axis along which to compute the index of the maximum
        #argmax Returns: the index of the maximum value along a given axis

        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
        cost = T.nnet.categorical_crossentropy(self.p_y_given_x, y).sum()
        return cost

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=10,
                           batch_size=600):

    train_set_x, train_set_y = load_data('train')

    train_set_x = theano.shared(numpy.asarray(train_set_x, dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX), borrow=True)

    #print train_set_x.shape
    #print len(train_set_y)
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size


    print('... building the model ...')
    #print 'n_test_batches : {}'.format(n_test_batches)
    #print 'batch_size: {}'.format(batch_size)
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.matrix('y') # real one-hot indexes
    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training the model')
    start_time = timeit.default_timer()

    epoch = 0
    while (epoch < n_epochs):#n_epochs=1000
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

    end_time = timeit.default_timer()
    print 'The code run for {} epochs, with {} epochs/sec'.format(
                            epoch, 1. * epoch / (end_time - start_time))
    
    with open('best_model.pkl', 'wb') as f:
            pickle.dump(classifier, f)

def test():
    classifier = pickle.load(open('best_model.pkl'))
    predict_model = theano.function(
            inputs=[classifier.input],
            outputs=classifier.y_pred)

    test_set_x, test_set_y = load_data('test')

    error_n = 0
    predicted_values = predict_model(test_set_x)
    for i in range(test_set_y.shape[0]):
        if test_set_y[i][predicted_values[i]] != 1:
            error_n += 1

    print "Predicted error rate: "
    print error_n*1.0 / test_set_y.shape[0]

if __name__ == '__main__':
    sgd_optimization_mnist()
    test()