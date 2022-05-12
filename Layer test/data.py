# data.py
from pennylane import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from qiskit_machine_learning.datasets import ad_hoc_data

# Collect this in a class
class Data:

	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def size(self):
		return len(self.Y)

# Split a data object into training and validation data
# p is the proportion of the data which should be used for training
#def split_data(data, p):
#
#	X_train, X_val, Y_train, Y_val = train_test_split(data.X, data.Y, train_size = p)
#
#	return Data(X_train, Y_train), Data(X_val, Y_val)

# Reduces the dimension of the features based on principal component analysis
def reduce_data(data, dim):
	# Reduce dimensions
	X_red = data.X
	pca = PCA(n_components = dim)
	X_red = pca.fit_transform(X_red)

	# PennyLane numpy differ from normal numpy.
	# Converts np.ndarray to pennylane.np.tensor.tensor
	X_red = np.array([np.array(x) for x in X_red])

	return Data(X_red, data.Y)

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

def load_data_forest(size = 500):

	# Load the data set
	data = fetch_covtype()

	X_raw = data['data']
	Y_raw = data['target']

	# Number of data points
	N = len(Y_raw)

	X_ones = np.array([np.array(X_raw[i]) for i in range(N) if Y_raw[i] == 1])
	X_twos = np.array([np.array(X_raw[i]) for i in range(N) if Y_raw[i] == 2])

	N_ones = len(X_ones)
	N_twos = len(X_twos)

	n = size // 2

	indexes_ones = np.random.choice(range(N_ones), size = n, replace = False)
	indexes_twos = np.random.choice(range(N_twos), size = n, replace = False)

	X_ones = X_ones[indexes_ones]
	X_twos = X_twos[indexes_twos]

	X = np.concatenate((X_ones, X_twos))
	Y = np.concatenate((np.full(n, -1), np.full(n, 1)))

	return Data(X, Y)

def load_data_adhoc(dimensions = 2, size = 500, gap = 0.3):

	# Load the data set
	X, Y, _, _ = ad_hoc_data(training_size = size, test_size = 0, n = dimensions, gap = gap)

	# PennyLane numpy differ from normal numpy.
	# Converts np.ndarray to pennylane.np.tensor.tensor
	Y = np.array([y[0] for y in Y]) # Y has two columns although these columns are either [0, 1] or [1, 0] so we can discard the second dimension
	X = np.array([np.array(x) for x in X], requires_grad = False)

	# Scale and translate Y from 0 and 1 to -1 and 1
	y = 2 * Y - 1

	return Data(X, Y)
