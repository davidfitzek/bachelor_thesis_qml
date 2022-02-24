# common.py

import numpy as np

# Labels, predictions are assumed to be of equal length
def square_loss(labels, preds):
    loss = sum((l - p) ** 2 for l, p in zip(labels, preds))
    return loss / len(labels)

# Labels, predictions are assumed to be of equal length
def accuracy(labels, preds):
    tol = 1e-5
    loss = sum((l - p) < tol for l, p in zip(labels, preds))
    return loss / len(labels)

# Splits data into train and validation data randomly
# percentage is a number between 0 and 1 indicating what percentage should be test data
def split_data(X, Y, percentage):
    n = len(Y)
    
    indexes = np.arange(n)
    
    indexes_train = np.random.choice(indexes, np.int64(percentage * n), replace = False)
    indexes_train = np.sort(indexes_train)
    indexes_val = np.setdiff1d(indexes, indexes_train, assume_unique = True)

    X_train = X[indexes_train]
    X_val = X[indexes_val]
    Y_train = Y[indexes_train]
    Y_val = Y[indexes_val]

    return X_train, X_val, Y_train, Y_val

