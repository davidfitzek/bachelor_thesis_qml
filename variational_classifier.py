# variational_classifier.py

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

n_wires = 4

dev = qml.device("default.qubit", wires = n_wires)

# The layer for the circuit
def layer(W):
    n = len(W)

    # Adds rotation matrices
    for i, row in enumerate(W):
        qml.Rot(row[0], row[1], row[2], wires = i)

    # Adds controlled NOT matrices
    for i in range(n):
        qml.CNOT(wires = [i, i % n])

# The state preparation for the circuit
def statepreparation(x):
    qml.BasisState(x, wires = [i for i in range(n_wires)])

# The circuit
@qml.qnode(dev)
def circuit(weights, x):

    statepreparation(x)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, x, bias):
    return circuit(weights, x) + bias

# Labels, predictions are assumed to be of equal length
def square_loss(labels, preds):
    loss = sum((l - p) ** 2 for l, p in zip(labels, preds))
    return loss / len(labels)

# Labels, predictions are assumed to be of equal length
def accuracy(labels, preds):
    tol = 1e-5
    loss = sum((l - p) < tol for l, p in zip(labels, preds))
    return loss / len(labels)

def cost(weights, bias, X, Y):
    preds = [variational_classifier(weights, x, bias) for x in X]
    return square_loss(Y, preds)
