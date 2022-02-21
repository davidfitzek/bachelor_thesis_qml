# iris_classifier.py

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

np.random.seed(123) # Set seed for reproducibility

n_wires = 2
n_qubits = n_wires

dev = qml.device("default.qubit", wires = 2)

# The layer for the circuit
def layer(W):
    n = len(W)

    # Adds rotation matrices
    for i, row in enumerate(W):
        qml.Rot(row[0], row[1], row[2], wires = i)

    # Adds controlled NOT matrices
    for i in range(n):
        qml.CNOT(wires = [i, (i + 1) % n])

# TODO: Understand what these five angles are
def get_angles(x):

    beta_0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta_1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta_2 = 2 * np.arcsin(np.sqrt(x[2] ** 2 + x[3] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2))

    return np.array([beta_2, - beta_1 / 2, beta_1 / 2, - beta_0 / 2, beta_0 / 2])

def statepreparation(angles):

    qml.RY(angles[0], wires = 0)

    for i in range(2):
        for j in range(2):
            qml.CNOT(wires = [0, 1])
            qml.RY(angles[2 * i + j + 1], wires = 1)
        qml.PauliX(wires = 0)

# The circuit
@qml.qnode(dev)
def circuit(weights, angles):

    statepreparation(angles)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, angles, bias):
    return circuit(weights, angles) + bias

# Labels, predictions are assumed to be of equal length
def square_loss(labels, preds):
    loss = sum((l - p) ** 2 for l, p in zip(labels, preds))
    return loss / len(labels)

def cost(weights, bias, features, labels):
    preds = [variational_classifier(weights, feature, bias) for feature in features]
    return square_loss(labels, preds)

# Load the data
data = np.loadtxt("data/iris_classes1and2_scaled.txt")
X = data[:, 0 : 2]
print("First X sample (original)  : {}".format(X[0]))

# Pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
print("First X sample (padded)    : {}".format(X_pad[0]))

# Normalise each input
norm = np.sqrt(np.sum(X_pad ** 2, -1))
X_normalised = (X_pad.T / norm).T
print("First X sample (normalised): {}".format(X_normalised[0]))

# Angles for state preparation are new features
features = np.array([get_angles(x) for x in X_normalised], requires_grad = False)
print("First features sample      :", features[0])

Y = data[:, -1]

