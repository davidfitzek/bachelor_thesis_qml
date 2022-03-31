# data.py
from pennylane import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
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
def split_data(data, p):

	X_train, X_val, Y_train, Y_val = train_test_split(data.X, data.Y, train_size = p)

	return Data(X_train, Y_train), Data(X_val, Y_val)

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

def load_data_forest():

	# Load the data set
	data = fetch_covtype()

	X_raw = data['data']
	Y_raw = data['target']

	Y = []
	X = []

	# It turns out this will yield a dataset of 495 141 points
	# We limit the size to the first 500 for now
	# Might be replaced with a random sample instead

	# In the forest data elements can be of type 1 to 7
	# We only log at type 1 and type 2
	cnt = 0
	for x, y in zip(X_raw, Y_raw):
		if y < 3:
			X.append(np.array(x))
			Y.append(y)
			# Not 
			cnt = cnt + 1
			if cnt == 500:
				break

	# Convert to numpy array
	Y = np.array(Y)
	X = np.array(X)

	# Scale and translate Y from 1 and 2 to -1 and 1
	Y = 2 * Y - 3

	return Data(X, Y)

def load_data_adhoc(dimensions = 2, size = 500, gap = 0.3):

	# Load the data set
	X, Y, _, _ = ad_hoc_data(training_size = size, test_size = 0, n = dimensions, gap = gap)

	# PennyLane numpy differ from normal numpy.
	# Converts np.ndarray to pennylane.np.tensor.tensor
	Y = np.array([y[0] for y in Y]) # Y has two columns although these columns are either [0, 1] or [1, 0] so we can discard the second dimension
	X = np.array([np.array(x) for x in X], requires_grad = False)

	return Data(X, Y)
