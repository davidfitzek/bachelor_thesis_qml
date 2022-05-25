# ibm_classifier.py

import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as opt

import common as com
import data as dat

import statistics as stat
import json

import sys

np.random.seed(123) # Set seed for reproducibility

circuit_calls = 0 # Global for counting number of times the circuit has been called

# The layer for the circuit
def layer_ex1(weights):
	n = len(weights)

	# Adds rotation matrices
	for i, row in enumerate(weights):
		qml.Rot(row[0], row[1], row[2], wires = i)

	# Adds controlled NOT matrices
	for i in range(n):
		qml.CNOT(wires = [i, i + 1 if i + 1 < n else 0])

def layer_ex2(weights):
	n = len(weights)

	# Adds rotation matrices and controlled NOT matrices
	for i, row in enumerate(weights):
		qml.Rot(row[0], row[1], row[2], wires = i)
		qml.CNOT(wires = [i, i + 1 if i + 1 < n else 0])

def stateprep_amplitude(features):
	wires = np.int64(np.ceil(np.log2(len(features))))
	# Normalise the features here and also pad it to have the length of a power of two
	qml.AmplitudeEmbedding(features = features, wires = range(wires), pad_with = 0, normalize = True)

def stateprep_Z(features):
	wires = len(features)
	for wire in range(wires):
		qml.Hadamard(wire)
	qml.AngleEmbedding(features = features, wires = range(wires), rotation = 'Z')

def stateprep_ZZ(features):
	wires = len(features)
	for wire in range(wires):
		qml.Hadamard(wire)
	qml.AngleEmbedding(features = features, wires = range(wires), rotation = 'Z')
	for i in range(1, wires):
		qml.CNOT(wires = [i - 1, i])
		qml.RY((np.pi - features[i - 1]) * (np.pi - features[i]))
		qml.CNOT(wires = [i - 1, i])

def stateprep_angle(features):
	wires = len(features)
	qml.AngleEmbedding(features = features, wires = range(wires), rotation = 'Y')

# The circuit function, allows variable statepreparation and layer functions
def circuit_fun(weights, features, stateprep_fun, layer_fun):
	global circuit_calls
	circuit_calls += 1

	stateprep_fun(features)

	for weight in weights:
		layer_fun(weight)

	return qml.expval(qml.PauliZ(0))

def variational_classifier_fun(weights, features, bias, circuit_fun):
	return circuit_fun(weights, features) + bias

def cost_fun(weights, bias, features, labels, variational_classifier_fun):
	preds = [variational_classifier_fun(weights, feature, bias) for feature in features]
	return com.square_loss(labels, preds)

def classify(weights, bias, data, data_train, data_val, circuit, cross_iter):
	# Variational classifier function used by pennylane
	def variational_classifier(weights, features, bias):
		return variational_classifier_fun(weights, features, bias, circuit)

	# Cost function used by pennylane
	def cost(weights, bias, features, labels):
		return cost_fun(weights, bias, features, labels, variational_classifier)

	# Number of training points, used when choosing batch indexes
	n_train = data_train.size()

	# Compute predictions on train and test set
	#predictions_train = [np.sign(variational_classifier(weights, x, bias)) for x in data_train.X]
	predictions_val = [np.sign(variational_classifier(weights, x, bias)) for x in data_val.X]

	# Compute accuracy on train and test set
	#accuracy_train = 0
	#accuracy_train = com.accuracy(data_train.Y, predictions_train)
	accuracy_train = 0
	accuracy_val = com.accuracy(data_val.Y, predictions_val)

	cost_ = cost(weights, bias, data.X, data.Y)

	print(
		'Cross validation iteration: {:5d} | Cost: {:0.7f} | Accuracy training: {:0.7f} | Accuracy validation: {:0.7f}'
		''.format(cross_iter + 1, cost_, accuracy_train, accuracy_val)
	)

	doc = {
		'cost': float(cost_),
		'acc_train': float(accuracy_train),
		'acc_val': float(accuracy_val),
	}

	return doc

def run_variational_classifier(param_file, n_qubits, n_layers, data, stateprep_fun, layer_fun, cross_fold):

	# Read in IBMQ token
	token = ''
	with open('ibmq_token', 'r') as f:
		token = f.read()[: -1] # Read in token and remove newline character

	# The device dused by pennylane
	#device = qml.device('default.qubit', wires = n_qubits)
	device = qml.device('qiskit.ibmq', wires = n_qubits, backend = 'ibmq_belem', ibmqx_token = token)

	# Circuit function used by pennylane
	@qml.qnode(device)
	def circuit(weights, x):
		return circuit_fun(weights, x, stateprep_fun, layer_fun)

	# Shuffle our data to introduce a random element to our train and test parts
	data = dat.shuffle_data(data)

	# Compute the size of 
	N = data.size()
	cross_size = N // cross_fold

	res = {} # dictionary for holding our accuracy results

	weights = 0
	bias = 0

	for cross_iter in range(cross_fold):

		# Read in the parameters from the file
		with open(param_file, 'r') as f:
			params = json.load(f)
			weights = np.array(params['cross iter' + str(cross_iter + 1)]['weights'])
			bias = np.array(params['cross iter' + str(cross_iter + 1)]['bias'], requires_grad = True)
			
		data_train, data_val = dat.split_data(data, cross_iter * cross_size, (cross_iter + 1) * cross_size)

		res['cross iter' + str(cross_iter + 1)] = classify(weights, bias, data, data_train, data_val, circuit, cross_iter)

	return res

def main():

	cross_fold = 10 # The ammount of parts the data is divided into, 1 gives no cross validation

	n_qubits = 2
	n_layers = 10

	# Can be any function that takes an input vector and encodes it
	stateprep_fun = stateprep_angle

	# Can be any function which takes in a matrix of weights and creates a layer
	layer_fun = layer_ex1

	# Load the data set
	data = dat.load_data_iris()
	#data = data.first(50)

	data = dat.reduce_data(data, n_qubits)
	#data = dat.scale_data(data, -1, 1)
	#data = dat.normalise_data(data)

	param_file = 'data/test_weights_bias_irispca_angle_bastlayers.json'

	res = run_variational_classifier(
		param_file,
		n_qubits,
		n_layers,
		data,
		stateprep_fun,
		layer_fun,
		cross_fold
	)
	
	# Dump data
	dump_file = 'data/test_irispca_angle_bastlayers_ibmq_belem.json'
	#dump_file = '/dev/null'
	with open(dump_file, 'w') as f:
		json.dump(res, f)
		print('Dumped data to ' + dump_file)

	# Compute some statistics with the accuracies
	final_acc = [val['acc_val'] for key, val in res.items()]

	mean = stat.mean(final_acc)
	stdev = stat.stdev(final_acc, xbar = mean)

	print('Final Accuracy: {:0.7f} +- {:0.7f}'.format(mean, stdev))
	print('Circuit Calls: {}'.format(circuit_calls))

if __name__ == '__main__':
	main()
