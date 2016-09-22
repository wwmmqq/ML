import theano
import theano.tensor as T

from data_helpers import load_data

def train(dataset='train', rec_model='lstm', optimizer='rmsprop',
	sample_length=200):

	print('train(..)')
	x = T.fmatrix('x')
	y = T.fmatrix('y')
	index = T.lscalar('index')
	n_x = 300
	n_y = 5
	n_h = 128

	learning_rate=0.001
	n_epochs = 100


	train_set_x, train_set_y= load_data(dataset)

	model = Lstm(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
		params=None)

	cost = model.cross_entropy(y)

	grads = T.grad(cost, model.params)
	updates = [(param, param - learning_rate * T.clip(grad, -grad_clip, grad_clip))
				for param, grad in zip(params, grads)
				]
	train_model = theano.function(
		inputs=[index],
		outputs=cost,
		givens={
			x: train_set_x[index],
			y: train_set_y[index]
		},
		updates=updates
	)

	n_train_examples = train_set_x.get_value(borrow=True).shape[0]
	epoch = 0
	while(epoch < n_epochs)
		epoch += 1
		for i in xrange(n_train_examples):
			train_cost = train_model(i)

	with open('./best_model.pkl', 'w') as f:
		pkl.dump(model, f, pkl.HIGHEST_PROTOCOL)

def load_model(self, path):
	with open(path, 'r') as f:
		classifier = pkl.load(f)

	predict_model = theano.function(
		inputs=[classifier.x],
		outputs=classifier.y)

	test_set_x, test_set_y = load_data('test')

	predicted_values = predict_model(test_set_x)
	error_n = 0
	for i in xrange(test_set_y.shape[0]):
		if test_set_y[i] != predicted_values[i]:
			error_n += 1

	print 'test error rate: {}'.format(error_n*1.0 / test_set_y.shape[0]);


