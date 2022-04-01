from sklearn import datasets
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.datasets import ad_hoc_data
from QKE_functions import *

def load_data_forest(n_data, n_classes, n_attributes):
    forest = datasets.fetch_covtype()
    data = forest
    X_raw = data.data
    Y_raw = data.target

    [X, Y] = reduceClassDimensions(X_raw, Y_raw, n_classes, n_data)

    forest_sample_train, forest_sample_test, forest_label_train, forest_label_test = train_test_split(
        X, Y, test_size=0.3, random_state=22)

    [forest_sample_train, forest_sample_test] = reduceAttributeDimensions(n_attributes, forest_sample_train, forest_sample_test)

    return forest_sample_train, forest_sample_test, forest_label_train, forest_label_test

def load_data_iris(n_data, n_classes):


    iris = datasets.load_iris()
    data = iris
    X_raw = data.data
    Y_raw = data.target
    n_classes = 2

    [X, Y] = reduceClassDimensions(X_raw, Y_raw, n_classes, n_data)

    iris_sample_train, iris_sample_test, iris_label_train, iris_label_test = train_test_split(
        X, Y, test_size=0.3, random_state=22)

    return iris_sample_train, iris_sample_test, iris_label_train, iris_label_test

def load_data_breast(n_attributes):
    breast = datasets.load_breast_cancer()
    breast_sample_train, breast_sample_test, breast_label_train, breast_label_test = train_test_split(
        breast.data, breast.target, test_size=0.3, random_state=22)

    [breast_sample_train, breast_sample_test] = reduceAttributeDimensions(n_attributes, breast_sample_train, breast_sample_test)

    return breast_sample_train, breast_sample_test, breast_label_train, breast_label_test

def load_data_digits(n_class_digits, n_attributes):
    digits = datasets.load_digits(n_class=n_class_digits)
    data = digits
    digits_sample_train, digits_sample_test, digits_label_train, digits_label_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=22)

    [digits_sample_train, digits_sample_test] = reduceAttributeDimensions(n_attributes, digits_sample_train, digits_sample_test)

    return digits_sample_train, digits_sample_test, digits_label_train, digits_label_test

def load_data_adhoc (n_data, adhoc_dimension):
    
    adhoc_sample_train, adhoc_sample_test, adhoc_label_train, adhoc_label_test, total = ad_hoc_data(
    training_size=20,#int(n_data*0.7),
    test_size=5,#int(n_data*0.3),
    n=2,
    gap=0.3,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
    )
    return adhoc_sample_train, adhoc_sample_test, adhoc_label_train, adhoc_label_test
