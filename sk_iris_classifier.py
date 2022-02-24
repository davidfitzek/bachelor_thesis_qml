# sk_iris_classifier.py

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn.datasets import load_iris

import common as com

np.random.seed(123) # Set seed for reproducibility

n_wires = 2
n_qubits = n_wires
n_layers = 4

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

def cost(weights, bias, features, labels):
    preds = [variational_classifier(weights, feature, bias) for feature in features]
    return com.square_loss(labels, preds)

data = load_iris()

features = data['data']
targets = data['target']

# We will only look at two types, -1 and 1
# In targets, elements are of three types 0, 1, and 2.
# We simply cutoff the 2:s for now
# The array is sorted so we can easily find first occurence of a 2 with binary search
cutoff = np.searchsorted(targets, 2)

# Now simply remove the targets and features corresponding to the 2:s
features = features[: cutoff]
targets = targets[: cutoff]

# Scale and translate targets from 0 and 1 to -1 and 1
targets = 2 * targets - 1

print(features)
print(targets)

