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

X = data['data']
Y = data['target']

# We will only look at two types, -1 and 1
# In Y, elements are of three types 0, 1, and 2.
# We simply cutoff the 2:s for now
# The array is sorted so we can easily find first occurence of a 2 with binary search
cutoff = np.searchsorted(Y, 2)

# Now simply remove the x:s and y:s corresponding to the 2:s
X = X[: cutoff]
Y = Y[: cutoff]

# Scale and translate Y from 0 and 1 to -1 and 1
Y = 2 * Y - 1

# Normalise each row in X
X_norm = np.linalg.norm(X, axis = 1).reshape(100, 1) # Because X is ndarray X_norm is a tensor 
X = X / X_norm

# Get the angles
X = np.array([get_angles(x) for x in X], requires_grad = False)

print('Data')
for x, y in zip(X, Y):
    print('\t' + '{}'.format(x).ljust(45) + ' : ' + '{}'.format(y).rjust(2))

# Percentage of the data which should be used for training
percentage = 0.7

# Split data into train and validation
X_train, X_val, Y_train, Y_val = com.split_data(X, Y, percentage)

n_train = len(Y_train)
n_val = len(Y_val)

weights_init = 0.01 * np.random.randn(n_layers , n_qubits, 3, requires_grad = True)
bias_init = np.array(0.0, requires_grad = True)

opt = NesterovMomentumOptimizer(0.01)
batch_size = 5

# train the variational classifier
weights = weights_init
bias = bias_init
n_steps = 60

for i in range(n_steps):

    # Update the weights by one optimiser step
    batch_index = np.random.randint(0, high = n_train, size = (batch_size, ))
    X_train_batch = X_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, X_train_batch, Y_train_batch)

    # Compute predictions on train and test set
    preds_train = [np.sign(variational_classifier(weights, x, bias)) for x in X_train]
    preds_val = [np.sign(variational_classifier(weights, x, bias)) for x in X_val]

    # Compute accuracy on train and test set
    acc_train = com.accuracy(Y_train, preds_train)
    acc_test = com.accuracy(Y_val, preds_val)

    print(
        'Iteration: {:5d} | Cost: {:0.7f} | Accuracy train: {:0.7f} | Accuracy test: {:0.7f} '
        ''.format(i + 1, cost(weights, bias, X, Y), accuracy_train, accuracy_test)
    )

