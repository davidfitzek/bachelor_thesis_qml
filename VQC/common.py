# common.py

import numpy as np

# Labels, predictions are assumed to be of equal length
def square_loss(labels, preds):
    loss = sum((l - p) ** 2 for l, p in zip(labels, preds))
    return loss / len(labels)

# Labels, predictions are assumed to be of equal length
def accuracy(labels, preds):
    tol = 1e-5 # Use a tolerance to determine if two values are "equal"
    loss = sum(abs(l - p) < tol for l, p in zip(labels, preds))
    return loss / len(labels)
