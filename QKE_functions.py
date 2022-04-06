from random import sample
from traceback import print_tb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale
import numpy as np

#This function reduces the amount of attributes of the dataset.
#This is done with principal component analysis (PCA).
def reduceAttributeDimensions(n_dim, sample_train, sample_test):
    # Reduce dimensions with PCA
    pca = PCA(n_components=n_dim).fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)
    # Normalise the data
    normalise(sample_train, sample_test)

    # Scale the data to (-1, 1)
    scale(sample_train, sample_test, -1, 1)

    return sample_train, sample_test

def normalise(sample_train, sample_test):
    # Normalise the data
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    return sample_train, sample_test

def scale(sample_train, sample_test, min, max):
    # Normalise the data
    digits_samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((min, max)).fit(digits_samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)

    return sample_train, sample_test

#This function reduces the amount of classes of the dataset.
#This is done by only selecting the labels we want to use.
def reduceClassDimensions(X_raw, Y_raw, n_classes, n_data):
    X = []
    Y = []

    cnt = 0
    for x, y in zip(X_raw, Y_raw):
        if y <= n_classes:
            X.append(np.array(x))
            Y.append(y)
            cnt = cnt + 1
            if cnt == n_data:
                break
    return X, Y