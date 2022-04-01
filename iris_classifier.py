# iris_classifier.py

import statistics

import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as opt

import common as com
import data as dat

import json

np.random.seed(123) # Set seed for reproducibility

#flytta in i data.py
# Collect this in a class
class Data:

	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def size(self):
		return len(self.Y)

# The layer for the circuit
def layer_ex1(weights):
	n = len(weights)

	# Adds rotation matrices
	for i, row in enumerate(weights):
		qml.Rot(row[0], row[1], row[2], wires = i)

	# Adds controlled NOT matrices
	for i in range(n):
		qml.CNOT(wires = [i, (i + 1) % n])

def layer_ex2(weights):
	n = len(weights)

	# Adds rotation matrices and controlled NOT matrices
	for i, row in enumerate(weights):
		qml.Rot(row[0], row[1], row[2], wires = i)
		qml.CNOT(wires = [i, (i + 1) % n])

def stateprep_amplitude(features):
	wires = np.int64(np.ceil(np.log2(len(features))))
	# Normalise the features here and also pad it to have the length of a power of two
	qml.AmplitudeEmbedding(features = features, wires = range(wires), pad_with = 0, normalize = True)

def stateprep_angle(features):
	wires = len(features)
	qml.AngleEmbedding(features = features, wires = range(wires), rotation = 'Y')

# The circuit function, allows variable statepreparation and layer functions
def circuit_fun(weights, features, stateprep_fun, layer_fun):

	stateprep_fun(features)

	for weight in weights:
		layer_fun(weight)

	return qml.expval(qml.PauliZ(0))

def variational_classifier_fun(weights, features, bias, circuit_fun):
	return circuit_fun(weights, features) + bias

def cost_fun(weights, bias, features, labels, variational_classifier_fun):
	preds = [variational_classifier_fun(weights, feature, bias) for feature in features]
	return com.square_loss(labels, preds)

def optimise(n_iter, weights, bias, data, data_train, data_val, circuit):
	optimiser = opt.NesterovMomentumOptimizer(stepsize = 0.01) # Performs much better than GradientDescentOptimizer
	#optimiser = opt.AdamOptimizer(stepsize = 0.01) # To be tried, was mentioned
	#optimiser = opt.GradientDescentOptimizer(stepsize = 0.01)
	batch_size = 5 # This might be something which can be adjusted

	costs = []
	acc_train = []
	acc_val = []

	# Variational classifier function used by pennylane
	def variational_classifier(weights, features, bias):
		return variational_classifier_fun(weights, features, bias, circuit)

	# Cost function used by pennylane
	def cost(weights, bias, features, labels):
		return cost_fun(weights, bias, features, labels, variational_classifier)

	# Number of training points, used when choosing batch indexes
	n_train = data_train.size()

	for i in range(n_iter):

		# Update the weights by one optimiser step
		batch_index = np.random.randint(0, high = n_train, size = (batch_size, ))
		X_train_batch = data_train.X[batch_index]
		Y_train_batch = data_train.Y[batch_index]
		weights, bias, _, _ = optimiser.step(cost, weights, bias, X_train_batch, Y_train_batch)
		# Compute predictions on train and test set
		predictions_train = [np.sign(variational_classifier(weights, x, bias)) for x in data_train.X]
		predictions_val = [np.sign(variational_classifier(weights, x, bias)) for x in data_val.X]

		# Compute accuracy on train and test set
		accuracy_train = com.accuracy(data_train.Y, predictions_train)
		accuracy_val = com.accuracy(data_val.Y, predictions_val)

		cost_ = cost(weights, bias, data.X, data.Y)

		print(
			'Cross validation iteration: {:5d} | Iteration: {:5d} | Cost: {:0.7f} | Accuracy train: {:0.7f} | Accuracy validation: {:0.7f}'
			''.format(cross_iter + 1, i + 1, cost(weights, bias, data.X, data.Y), accuracy_train, accuracy_val)
		)

		costs.append(float(cost_))
		acc_train.append(float(accuracy_train))
		acc_val.append(float(accuracy_val))

	doc = {
		'costs': costs,
		'acc_train': acc_train,
		'acc_val': acc_val
	}

	with open('data/test.json', 'w') as f:
		json.dump(doc, f)
	#returns a final accuracy
	return accuracy_val


#flytta till data.py
# Split a data object into training and validation data
# start and stop are the indicies for where that validation data begins and ends
# The rest of the data points are assumed to be training data
def split_data(data, start, stop):

	X_train = np.concatenate((data.X[ : start], data.X[stop : ]))
	Y_train = np.concatenate((data.Y[ : start], data.Y[stop : ]))

	X_val = data.X[start : stop]
	Y_val = data.Y[start : stop]

	return Data(X_train, Y_train), Data(X_val, Y_val)

# Shuffles the data points
def shuffle_data(data):
	N = data.size() # Number of data points

	indexes = np.random.permutation(N)

	return Data(data.X[indexes], data.Y[indexes])

def run_variational_classifier(n_iter, n_qubits, n_layers, data, stateprep_fun, layer_fun, cross_fold):

	device = qml.device("default.qubit", wires = n_qubits)

	# Circuit function used by pennylane
	@qml.qnode(device)
	def circuit(weights, x):
		return circuit_fun(weights, x, stateprep_fun, layer_fun)

	# Shuffle our data to introduce a random element to our train and test parts
	data = shuffle_data(data)

	# Compute the size of
	N = data.size()
	cross_size = N // cross_fold

	res = [] # List for holding our accuracy results

	for cross_iter in range(cross_fold):

		data_train, data_val = split_data(data, cross_iter * cross_size, (cross_iter + 1) * cross_size)

		weights = 0.01 * np.random.randn(n_layers , n_qubits, 3, requires_grad = True) # Initial value for the weights
		bias = np.array(0.0, requires_grad = True) # Initial value for the bias

		res.append(
			optimise(n_iter, weights, bias, data, data_train, data_val, circuit, cross_iter)
		)
	return res

def main():
	n_iter = 2 # Number of iterations, should be changed to a tolerance based process instead
	n_qubits = 2
	n_layers = 6

	# Can be any function that takes an input vector and encodes it
	stateprep_fun = stateprep_amplitude

	# Can be any function which takes in a matrix of weights and creates a layer
	layer_fun = layer_ex1

	cross_fold = 2  # The ammount of parts the data is divided into, =1 gives no cross validation

	# Load the iris data
	data = dat.load_data_cancer()
	data = dat.reduce_data(data, n_qubits)

	res = run_variational_classifier(
		n_iter,
		n_qubits,
		n_layers,
		data,
		stateprep_fun,
		layer_fun,
		cross_fold
		)

	# Convert numpy tensors to floats
	res = [float(r) for r in res]

	print('Final Accuracy: {:0.7f} +- {:0.7f}'.format(statistics.mean(res), statistics.stdev(res)))

if __name__ == '__main__':
	main()
