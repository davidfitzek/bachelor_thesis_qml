# iris_classifier.py

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

import common as com

import matplotlib.pyplot as plt

import time

np.random.seed(123) # Set seed for reproducibility

# Collect this in a class
class Data:

	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

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

def optimise(accuracy_stop, cost_stop, iter_stop, weights, bias, data, data_train, data_val, circuit, n_layers):
	#opt = NesterovMomentumOptimizer(stepsize = 0.01) # Performs much better than GradientDescentOptimizer
	opt = AdamOptimizer(stepsize = 0.01) # To be tried, was mentioned
	batch_size = 5 # This might be something which can be adjusted
	
	# Variational classifier function used by pennylane
	def variational_classifier(weights, features, bias):
		return variational_classifier_fun(weights, features, bias, circuit)

	# Cost function used by pennylane
	def cost(weights, bias, features, labels):
		return cost_fun(weights, bias, features, labels, variational_classifier)

	# Number of training points, used when choosing batch indexes
	n_train = len(data_train.Y)

	accuracy_val = 0.0
	cost_var = 100 #just something big
	i = 0
	while i < iter_stop and ((accuracy_val < accuracy_stop) or (cost_var > cost_stop)):
		# Update the weights by one optimiser step
		batch_index = np.random.randint(0, high = n_train, size = (batch_size, ))
		X_train_batch = data_train.X[batch_index]
		Y_train_batch = data_train.Y[batch_index]
		weights, bias, _, _ = opt.step(cost, weights, bias, X_train_batch, Y_train_batch)
		# Compute predictions on train and test set
		predictions_train = [np.sign(variational_classifier(weights, x, bias)) for x in data_train.X]
		predictions_val = [np.sign(variational_classifier(weights, x, bias)) for x in data_val.X]

		# Compute accuracy on train and test set
		accuracy_train = com.accuracy(data_train.Y, predictions_train)
		accuracy_val = com.accuracy(data_val.Y, predictions_val)

		cost_var = float(cost(weights, bias, data.X, data.Y))

		print(
			'Iteration: {:5d} | Cost: {:0.7f} | Accuracy train: {:0.7f} | Accuracy validation: {:0.7f} | Layers: {:d} '
			''.format(i + 1, cost_var, accuracy_train, accuracy_val, n_layers))

		i += 1

	return [i, cost_var]

# Split a data object into training and validation data
# p is the proportion of the data which should be used for training
def split_data(data, p):

	X_train, X_val, Y_train, Y_val = train_test_split(data.X, data.Y, train_size = p)

	return Data(X_train, Y_train), Data(X_val, Y_val)

def run_variational_classifier(n_qubits, n_layers, data, stateprep_fun, layer_fun, accuracy_stop, cost_stop, iter_stop):

	# The device and qnode used by pennylane
	device = qml.device("default.qubit", wires = n_qubits)

	# Circuit function used by pennylane
	@qml.qnode(device)
	def circuit(weights, x):
		return circuit_fun(weights, x, stateprep_fun, layer_fun)

	# The proportion of the data which should be use for training
	p = 0.7

	data_train, data_val = split_data(data, p)

	weights = 0.01 * np.random.randn(n_layers , n_qubits, 3, requires_grad = True) # Initial value for the weights
	bias = np.array(0.0, requires_grad = True) # Initial value for the bias

	return optimise(accuracy_stop, cost_stop, iter_stop, weights, bias, data, data_train, data_val, circuit, n_layers)

# Load the iris data set from sklearn into a data object
def load_data_iris():

	# Load the data set
	data = load_iris()

	X = data['data']
	Y = data['target']

	# We will only look at two types, -1 and 1
	# In Y, elements are of three types 0, 1, and 2.
	# We simply cutoff the 2:s for now
	# The array is sorted so we can easily find first occurence of a 2 with binary search
	cutoff = np.searchsorted(Y, 2)

	# Now simply remove the x:s and y:s corresponding to the 2:s
	X = X[: cutoff]
	Y = Y[: cutoff]

	# Scale and translate Y from 0 and 1 to -1 and 1
	Y = 2 * Y - 1
	Y = np.array(Y) # PennyLane numpy differ from normal numpy. Converts np.ndarray to pennylane.np.tensor.tensor

	# PennyLane numpy differ from normal numpy.
	# Converts np.ndarray to pennylane.np.tensor.tensor
	Y = np.array(Y)
	X = np.array([np.array(x) for x in X], requires_grad = False)

	return Data(X, Y)

def load_data_cancer():

	# Load the data set
	data = load_breast_cancer()

	X = data['data']
	Y = data['target']

	# Scale and translate Y from 0 and 1 to -1 and 1
	Y = 2 * Y - 1

	# PennyLane numpy differ from normal numpy.
	# Converts np.ndarray to pennylane.np.tensor.tensor
	Y = np.array(Y)
	X = np.array([np.array(x) for x in X], requires_grad = False)

	return Data(X, Y)

def main():

	n_qubits = 2
	#it will test all the number of layers up to this number
	range_layers = 25

	# if the accuracy validation is higher and the cost is lower or if the iterations are higher it stops
	accuracy_stop = 0.99
	cost_stop = 0.7
	iter_stop = 250

	# Can be any function that takes an input vector and encodes it
	stateprep_fun = stateprep_amplitude

	# Can be any function which takes in a matrix of weights and creates a layer
	layer_fun = layer_ex1

	# Load the iris data
	data = load_data_iris()

	#testing how many layers it takes to achieve accuracy_stop and cost_stop
	iterations = [0]*range_layers
	cost = [0]*range_layers
	sec = [0]*range_layers
	for i in range(range_layers):
		n_layers = i + 1
		print("Starting with layer " + str(n_layers) + " of " + str(range_layers))
		tic = time.perf_counter()
		[iterations[i], cost[i]] = run_variational_classifier(n_qubits, n_layers, data, stateprep_fun, layer_fun, accuracy_stop, cost_stop, iter_stop)
		toc = time.perf_counter()
		sec[i] = toc - tic

	plt.subplot(3, 1, 1)
	plt.plot(iterations)
	plt.xlabel("Layer")
	plt.ylabel("Iterations")

	plt.subplot(3, 1, 2)
	plt.plot(cost)
	plt.xlabel("Layer")
	plt.ylabel("Cost")

	plt.subplot(3, 1, 3)
	plt.plot(sec)
	plt.xlabel("Layer")
	plt.ylabel("Seconds to execute")

	plt.show()

if __name__ == '__main__':
	main()
