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
    indexes = np.random.choice(range(n), np.int64(percentage * n))

    X_train = X[indexes]
    X_val = X[~indexes]
    Y_train = Y[indexes]
    Y_val = Y[~indexes]

    return X_train, X_val, Y_train, Y_val

