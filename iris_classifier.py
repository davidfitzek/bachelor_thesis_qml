# iris_classifier.py

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

np.random.seed(123) # Set seed for reproducibility

n_wires = 2
n_qubits = n_wires
n_layers = 6

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

# Splits data into train and test data randomly
# percentile is a number between 0 and 1 indicating what percentage should be test data
def split_data(X, Y, percentage):
    n = len(Y)
    indexes = np.random.choice(range(n), np.int64(percentage * n))

    X_train = X[indexes]
    X_test = X[~indexes]
    Y_train = Y[indexes]
    Y_test = Y[~indexes]

    return X_train, X_test, Y_train, Y_test

# Load the data
data = np.loadtxt('data/iris_classes1and2_scaled.txt')
X = data[:, 0 : 2]
print('First X sample (original)'.ljust(28) + ': {}'.format(X[0]))

# Pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
print('First X sample (padded)'.ljust(28) + ': {}'.format(X_pad[0]))

# Normalise each input
norm = np.sqrt(np.sum(X_pad ** 2, -1))
X_normalised = (X_pad.T / norm).T
print('First X sample (normalised)'.ljust(28) + ': {}'.format(X_normalised[0]))

# Angles for state preparation are new features
features = np.array([get_angles(x) for x in X_normalised], requires_grad = False)
print('First features sample'.ljust(28) + ': {}'.format(features[0]))

Y = data[:, -1]

np.random.seed(123) # Set seed again for reproducibility

# Split data into train data and test data
features_train, features_test, Y_train, Y_test = split_data(features, Y, 0.7)

n_train = len(Y_train)
n_test = len(Y_test)

print('Number of train data points : {}'.format(n_train))
print('Number of test data points  : {}'.format(n_test))

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
    features_train_batch = features_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, features_train_batch, Y_train_batch)

    # Compute predictions on train and test set
    predictions_train = [np.sign(variational_classifier(weights, feature, bias)) for feature in features_train]
    predictions_test = [np.sign(variational_classifier(weights, feature, bias)) for feature in features_test]

    # Compute accuracy on train and test set
    accuracy_train = accuracy(Y_train, predictions_train)
    accuracy_test = accuracy(Y_test, predictions_test)

    print(
        'Iteration: {:5d} | Cost: {:0.7f} | Accuracy train: {:0.7f} | Accuracy test: {:0.7f} '
        ''.format(i + 1, cost(weights, bias, features, Y), accuracy_train, accuracy_test)
    )

