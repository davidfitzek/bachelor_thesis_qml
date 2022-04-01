# iris_classifier.py

import statistics

import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as opt

import common as com
import data as dat

import matplotlib.pyplot as plt

import time

import csv

np.random.seed(123) # Set seed for reproducibility


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

def optimise(accuracy_stop, cost_stop, iter_stop, weights, bias, data, data_train, data_val, circuit, n_layers, cross_iter):
	optimiser = opt.NesterovMomentumOptimizer(stepsize = 0.01) # Performs much better than GradientDescentOptimizer
	#optimiser = opt.AdamOptimizer(stepsize = 0.01) # To be tried, was mentioned
	#optimiser = opt.GradientDescentOptimizer(stepsize = 0.01)
	batch_size = 5 # This might be something which can be adjusted
	
	# Variational classifier function used by pennylane
	def variational_classifier(weights, features, bias):
		return variational_classifier_fun(weights, features, bias, circuit)

	# Cost function used by pennylane
	def cost(weights, bias, features, labels):
		return cost_fun(weights, bias, features, labels, variational_classifier)

	# Number of training points, used when choosing batch indexes
	n_train = data_train.size()

	accuracy_val = 0.0
	cost_var = 100 #just something big
	i = 0
	while i < iter_stop and ((accuracy_val < accuracy_stop) or (cost_var > cost_stop)):
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

		cost_var = float(cost(weights, bias, data.X, data.Y))

		print(
			'Cross validation iteration: {:d} | Iteration: {:d} | Cost: {:0.7f} | Accuracy train: {:0.7f} | Accuracy validation: {:0.7f} | Layers: {:d} '
			''.format(cross_iter + 1, i + 1, cost_var, accuracy_train, accuracy_val, n_layers))

		i += 1

	return [i, cost_var, float(accuracy_val)]
	#return accuracy_val
# Shuffles the data points
def shuffle_data(data):
	N = data.size() # Number of data points

	indexes = np.random.permutation(N)

	return dat.Data(data.X[indexes], data.Y[indexes])

def run_variational_classifier(cross_fold, n_qubits, n_layers, data, stateprep_fun, layer_fun, accuracy_stop, cost_stop, iter_stop):

	# The device and qnode used by pennylane
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

	res = []  # List for holding our accuracy results
	iteration_res = []
	cost_res = []
	accuracy_res = []

	for cross_iter in range(cross_fold):
		data_train, data_val = dat.split_data(data, cross_iter * cross_size, (cross_iter + 1) * cross_size)

		weights = 0.01 * np.random.randn(n_layers, n_qubits, 3, requires_grad=True)  # Initial value for the weights
		bias = np.array(0.0, requires_grad=True)  # Initial value for the bias
		res = optimise(accuracy_stop, cost_stop, iter_stop, weights, bias, data, data_train, data_val, circuit, n_layers, cross_iter)
		iteration_res.append(res[0])
		cost_res.append(res[1])
		accuracy_res.append(res[2])
	return [iteration_res, cost_res, accuracy_res]

def main():

	n_qubits = 2
	#n_qubits = 5

	#it will test all the number of layers up to this number
	range_layers = 15

	# if the accuracy validation is higher and the cost is lower or if the iterations are higher it stops
	accuracy_stop = 0.95
	cost_stop = 0.8
	iter_stop = 150

	cross_fold = 10  # The amount of parts the data is divided into, =1 gives no cross validation

	# Can be any function that takes an input vector and encodes it
	stateprep_fun = stateprep_amplitude

	# Can be any function which takes in a matrix of weights and creates a layer
	layer_fun = layer_ex1

	# Load data
	data = dat.load_data_iris()
	#data = dat.load_data_cancer()

	#testing how many layers it takes to achieve accuracy_stop and cost_stop
	iterations = [0]*range_layers
	cost = [0]*range_layers
	accuracy = [0]*range_layers
	sec = [0]*range_layers

	start_time = time.perf_counter()

	for i in range(range_layers):
		n_layers = i + 1
		print("Starting with layer " + str(n_layers) + " of " + str(range_layers))
		print(" ")
		tic = time.perf_counter()
		res = run_variational_classifier(cross_fold, n_qubits, n_layers, data, stateprep_fun, layer_fun, accuracy_stop, cost_stop, iter_stop)
		toc = time.perf_counter()

		iteration_res = res[0]
		cost_res = res[1]
		accuracy_res = res[2]

		iterations[i] = [statistics.mean(iteration_res), statistics.stdev(iteration_res)]
		cost[i] = [statistics.mean(cost_res), statistics.stdev(cost_res)]
		accuracy[i] = [statistics.mean(accuracy_res), statistics.stdev(accuracy_res)]
		sec[i] = toc - tic

		print(" ")
		print("It took " + str(sec[i]) + " seconds"  )
		print('Final Accuracy: {:0.7f} +- {:0.7f}'.format(statistics.mean(accuracy_res), statistics.stdev(accuracy_res)))
		print('Final Cost: {:0.7f} +- {:0.7f}'.format(statistics.mean(cost_res), statistics.stdev(cost_res)))
		print(" ")

	stop_time = time.perf_counter()
	total_time = (stop_time - start_time)/3600.0

	print("Done! It took " + str(total_time) + " hours to run this programme.")

	with open("iterarions.csv", "w") as f:
		write = csv.writer(f)

		write.writerow(iterations)

	with open("cost.csv", "w") as f:
		write = csv.writer(f)

		write.writerow(cost)

	with open("sec.csv", "w") as f:
		write = csv.writer(f)

		write.writerow(sec)

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
