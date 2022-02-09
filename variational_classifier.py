# variational_classifier.py

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

np.random.seed(123) # Set seed for reproducibility

n_wires = 4
n_qubits = 4
n_layers = 4

dev = qml.device("default.qubit", wires = n_wires)

# The layer for the circuit
def layer(W):
    n = len(W)

    # Adds rotation matrices
    for i, row in enumerate(W):
        qml.Rot(row[0], row[1], row[2], wires = i)

    # Adds controlled NOT matrices
    for i in range(n):
        qml.CNOT(wires = [i, (i + 1) % n])

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

data = np.loadtxt('data/parity.txt')
X = np.array(data[:, :-1], requires_grad = False)
Y = np.array(data[:, -1], requires_grad = False)
n = len(Y) # Number of data points
Y = 2 * Y - np.ones(n) # shift labels from {0, 1} to {-1, 1}

for x, y in zip(X, Y):
    print('X = {}, Y = {: d}'.format(x, int(y)))

weights_init = 0.01 * np.random.randn(n_layers, n_qubits, 3, requires_grad = True)
bias_init = np.array(0.0, requires_grad = True)

print('Initial weights and bias')
print(weights_init, bias_init)

opt = NesterovMomentumOptimizer(0.5)
batch_size = n

weights = weights_init
bias = bias_init
for i in range(20):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, len(X), (batch_size, ))
    X_batch = X[batch_index]
    Y_batch = Y[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

    # Compute accuracy
    predictions = [np.sign(variational_classifier(weights, x, bias)) for x in X]
    acc = accuracy(Y, predictions)

    print(
        'Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} '.format(
            i + 1, cost(weights, bias, X, Y), acc
        )
    )

