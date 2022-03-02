# iris_classifier.py

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

import common as com

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

# Looking at equation 8 in https://arxiv.org/pdf/quant-ph/0407010.pdf
# With n = number of qubits, and k = 1, 2, ..., n
def get_angles(x):

    # Number of qubits needed to encode x
    # Should be equal to n_qubits
    # Should be a power of two
    n = np.int64(np.ceil(np.log2(len(x))))

    # Matrix for holding our angles
    beta = np.zeros(shape = (2 ** (n - 1), n))

    for k in range(n):
        for j in range(2 ** (n - k - 1)):
            # Compute the numerator inside the arcsin
            num = np.sqrt(sum(
                np.abs(x[(2 * j + 1) * 2 ** k + l]) ** 2
                    for l in range(2 ** k)
            ))
            # Compute the denomenator inside the arcsin
            den = np.sqrt(sum(
                np.abs(x[j * 2 ** (k + 1) + l]) ** 2
                    for l in range(2 ** (k + 1))
            ))
            beta[j, k] = 2 * np.arcsin(num / den)

    #return beta
    return np.array([
        beta[0, 1],
        -beta[1, 1] / 2,
        beta[1, 1] / 2,
        -beta[0, 0] / 2,
        beta[0, 0] / 2
    ])

def statepreparation(angles):

    qml.RY(angles[0], wires = 0)

    # Should be the same as n_qubits
    n = len(angles) // 2

    for i in range(n):
        for j in range(n):
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

def optimise(n_iter, X_train, X_val, Y_train, Y_val):

    n_train = len(Y_train)

    weights_init = 0.01 * np.random.randn(n_layers , n_qubits, 3, requires_grad = True)
    bias_init = np.array(0.0, requires_grad = True)

    opt = NesterovMomentumOptimizer(0.01)
    batch_size = 5

    # train the variational classifier
    weights = weights_init
    bias = bias_init

    for i in range(n_iter):
    
        # Update the weights by one optimiser step
        batch_index = np.random.randint(0, high = n_train, size = (batch_size, ))
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_train_batch, Y_train_batch)

        # Compute predictions on train and test set
        predictions_train = [np.sign(variational_classifier(weights, x, bias)) for x in X_train]
        predictions_val = [np.sign(variational_classifier(weights, x, bias)) for x in X_val]

        # Compute accuracy on train and test set
        accuracy_train = com.accuracy(Y_train, predictions_train)
        accuracy_val = com.accuracy(Y_val, predictions_val)

        print(
            'Iteration: {:5d} | Cost: {:0.7f} | Accuracy train: {:0.7f} | Accuracy validation: {:0.7f} '
            ''.format(i + 1, cost(weights, bias, features, Y), accuracy_train, accuracy_val)
        )

# Load the data
data = np.loadtxt('data/iris_classes1and2_scaled.txt')
X = data[:, 0 : 2]
#print('First X sample (original)'.ljust(28) + ': {}'.format(X[0]))

# Pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
#print('First X sample (padded)'.ljust(28) + ': {}'.format(X_pad[0]))

# Normalise each input
norm = np.linalg.norm(X_pad)
X_normalised = (X_pad.T / norm).T
#print('First X sample (normalised)'.ljust(28) + ': {}'.format(X_normalised[0]))

# Angles for state preparation are new features
features = np.array([get_angles(x) for x in X_normalised], requires_grad = False)
#print('First features sample'.ljust(28) + ': {}'.format(features[0]))

Y = data[:, -1]

np.random.seed(123) # Set seed again for reproducibility

# Split data into train data and test data
features_train, features_val, Y_train, Y_val = com.split_data(features, Y, 0.7)

# Optimise the weights
optimise(60, features_train, features_val, Y_train, Y_val)
