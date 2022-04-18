from sklearn import datasets
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.datasets import ad_hoc_data, sample_ad_hoc_data
from QKE_functions import *

# Collect this in a class
class Data:

	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def size(self):
		return len(self.Y)

def load_data_forest_oscar(size = 500):

	# Load the data set
	data = datasets.fetch_covtype()

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


def load_data_forest(n_data, n_attributes):
    '''
    Loads the dataset forest with length n_data and returns four datasets
    [iris_sample_train, iris_sample_test, iris_label_train, iris_label_test]
    '''
    forest = datasets.fetch_covtype()
    data = forest
    X_raw = data.data
    Y_raw = data.target
    #The forest dataset has classes from (1-7), by setting 
    #n_classes = 2, we only look at 1 and 2.
    n_classes = 2

    #Reduces the amount of classes
    [X, Y] = reduceClassDimensions(X_raw, Y_raw, n_classes, n_data)

    #Split the dataset into training- and testset 
    #for the sample and label.
    forest_sample_train, forest_sample_test, forest_label_train, forest_label_test = train_test_split(
        X, Y, test_size=0.3, random_state=22)
    #Reduces the amount of attrubutes/features.
    [forest_sample_train, forest_sample_test] = reduceAttributeDimensions(n_attributes, forest_sample_train, forest_sample_test)

    return forest_sample_train, forest_sample_test, forest_label_train, forest_label_test

def load_data_iris(n_data):
    '''
    Loads the dataset iris with length n_data and returns four datasets
    [iris_sample_train, iris_sample_test, iris_label_train, iris_label_test]
    '''
    iris = datasets.load_iris()
    data = iris
    X_raw = data.data
    Y_raw = data.target
    #The forest dataset has classes from (0-2), by setting 
    #n_classes = 1, we only look at 0 and 1.
    n_classes = 1

    #Reduces the amount of classes
    [X, Y] = reduceClassDimensions(X_raw, Y_raw, n_classes, n_data)
    #Split the dataset into training- and testset 
    #for the sample and label.
    iris_sample_train, iris_sample_test, iris_label_train, iris_label_test = train_test_split(
        X, Y, test_size=0.3, random_state=22)

    return iris_sample_train, iris_sample_test, iris_label_train, iris_label_test

def load_data_breast(n_attributes, n_data):
    '''
    Loads the dataset breast_cancer and returns four datasets
    [breast_sample_train, breast_sample_test, breast_label_train, breast_label_test]
    '''
    breast = datasets.load_breast_cancer()
    data = breast
    X_raw = data.data
    Y_raw = data.target
    #The forest dataset has classes from (0-2), by setting 
    #n_classes = 1, we only look at 0 and 1.
    n_classes = 1

    #Reduces the amount of classes
    [X, Y] = reduceClassDimensions(X_raw, Y_raw, n_classes, n_data)
    
    #Split the dataset into training- and testset 
    #for the sample and label.
    breast_sample_train, breast_sample_test, breast_label_train, breast_label_test = train_test_split(
        X, Y, test_size=0.3, random_state=22)
    #Reduces the amount of attrubutes/features.
    [breast_sample_train, breast_sample_test] = reduceAttributeDimensions(n_attributes, breast_sample_train, breast_sample_test)

    return breast_sample_train, breast_sample_test, breast_label_train, breast_label_test

def load_data_digits(n_class_digits, n_attributes):
    '''
    Loads the dataset digits with n_class_digits digits and returns four datasets
    [breast_sample_train, breast_sample_test, breast_label_train, breast_label_test]
    '''
    digits = datasets.load_digits(n_class=n_class_digits)

    #Split the dataset into training- and testset 
    #for the sample and label.
    digits_sample_train, digits_sample_test, digits_label_train, digits_label_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=22)
    #Reduces the amount of attrubutes/features.
    [digits_sample_train, digits_sample_test] = reduceAttributeDimensions(n_attributes, digits_sample_train, digits_sample_test)

    return digits_sample_train, digits_sample_test, digits_label_train, digits_label_test

def load_data_adhoc (n_data, adhoc_dimension):
    
    adhoc_sample_train, adhoc_sample_test, adhoc_label_train, adhoc_label_test = ad_hoc_data(
    training_size=int(n_data*0.7),
    test_size=int(n_data*0.3),
    n=adhoc_dimension,
    gap=0.3,
    plot_data=False,
    one_hot=False,
    include_sample_total=False,
    )
    return adhoc_sample_train, adhoc_label_train, adhoc_sample_test, adhoc_label_test
