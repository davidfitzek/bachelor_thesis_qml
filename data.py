from sklearn import datasets
from sklearn.model_selection import train_test_split
from QKE_functions import *

def load_data_forest(n_data):
    # Forest dataset
    forest = datasets.fetch_covtype()
    data = forest
    X_raw = data.data
    Y_raw = data.target
    n_classes = 2
    [X, Y] = reduceClassDimensions(X_raw, Y_raw, n_classes, n_data)

    forest_sample_train, forest_sample_test, forest_label_train, forest_label_test = train_test_split(
        X, Y, test_size=0.3, random_state=22)

    [forest_sample_train, forest_sample_test] = reduceAttributeDimensions(4, forest_sample_train, forest_sample_test)

    return forest_sample_train, forest_sample_test, forest_label_train, forest_label_test

def load_data_iris(n_data):
    # Iris dataset
    iris = datasets.load_iris()
    data = iris
    X_raw = data.data
    Y_raw = data.target
    n_classes = 2
    [X, Y] = reduceClassDimensions(X_raw, Y_raw, n_classes, n_data)

    iris_sample_train, iris_sample_test, iris_label_train, iris_label_test = train_test_split(
        X, Y, test_size=0.3, random_state=22)

    return iris_sample_train, iris_sample_test, iris_label_train, iris_label_test

def load_data_breast():
    # Breastcancer dataset
    breast = datasets.load_breast_cancer()
    data = breast
    breast_sample_train, breast_sample_test, breast_label_train, breast_label_test = train_test_split(
        breast.data, breast.target, test_size=0.3, random_state=22)

    [breast_sample_train, breast_sample_test] = reduceAttributeDimensions(4, breast_sample_train, breast_sample_test)

    return breast_sample_train, breast_sample_test, breast_label_train, breast_label_test

def load_data_digits(n_class_digits):
    # Digits dataset
    digits = datasets.load_digits(n_class=n_class_digits)
    data = digits
    digits_sample_train, digits_sample_test, digits_label_train, digits_label_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=22)

    [digits_sample_train, digits_sample_test] = reduceAttributeDimensions(4, digits_sample_train, digits_sample_test)

    return digits_sample_train, digits_sample_test, digits_label_train, digits_label_test