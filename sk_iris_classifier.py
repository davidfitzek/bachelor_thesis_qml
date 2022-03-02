# sk_iris_classifier.py

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn.datasets import load_iris

import json
import lzma

import common as com

np.random.seed(123) # Set seed for reproducibility

n_wires = 2
n_qubits = n_wires
n_layers = 6

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

def optimise(n_iter, X, X_train, X_val, Y, Y_train, Y_val):

    n_train = len(Y_train)

    weights_init = 0.01 * np.random.randn(n_layers , n_qubits, 3, requires_grad = True)
    bias_init = np.array(0.0, requires_grad = True)

    opt = NesterovMomentumOptimizer(0.01)
    batch_size = 5

    # train the variational classifier
    weights = weights_init
    bias = bias_init

    # Save cost and accuracy in each iteration
    costs = np.zeros(n_iter)
    acc_train = np.zeros(n_iter)
    acc_val = np.zeros(n_iter)

    for i in range(n_iter):

        # Update the weights by one optimiser step
        batch_index = np.random.randint(0, high = n_train, size = (batch_size, ))
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_train_batch, Y_train_batch)

        # Compute predictions on train and test set
        preds_train = [np.sign(variational_classifier(weights, x, bias)) for x in X_train]
        preds_val = [np.sign(variational_classifier(weights, x, bias)) for x in X_val]

        # Compute accuracy on train and test set
        acc_train[i] = com.accuracy(Y_train, preds_train)
        acc_val[i] = com.accuracy(Y_val, preds_val)

        costs[i] = cost(weights, bias, X, Y)

        print(
            'Iteration: {:5d} | Cost: {:0.7f} | Accuracy train: {:0.7f} | Accuracy validation: {:0.7f} '
            ''.format(i + 1, costs[i], acc_train[i], acc_val[i])
        )

        #dump(weights, bias, X_train, X_val, Y_train, Y_val, costs, acc_train, acc_val)

def dump(weights, bias, X_train, X_val, Y_train, Y_val, costs, acc_train, acc_val):
    # Dumping result to json
    doc = {
        'weights': weights.tolist(),
        'bias': bias.tolist(),
        'X_train': X_train.tolist(),
        'X_val': X_val.tolist(),
        'Y_train': Y_train.tolist(),
        'Y_val': Y_val.tolist(),
        'costs': costs.tolist(),
        'acc_train': acc_train.tolist(),
        'acc_val': acc_val.tolist()
    }

    filename = 'data/sk_iris_result.json.xz'
    filedata = bytes(json.dumps(doc), encoding = 'utf8')

    with lzma.open(filename, 'w') as f:
        f.write(filedata)

    print('Dumped data to: {}'.format(filename))

# Move everything to main function
def main():
    # Load the data set
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
    Y = np.array(Y) # PennyLane numpy differ from normal numpy. Converts np.ndarray to pennylane.np.tensor.tensor

    # Normalise each row in X
    X_norm = np.linalg.norm(X, axis = 1).reshape(100, 1) # Because X is ndarray X_norm is a tensor 
    X = X / X_norm

    # Get the angles
    X = np.array([get_angles(x) for x in X], requires_grad = False)

    print('Data:')
    for x, y in zip(X, Y):
        print('\t' + '{}'.format(x).ljust(45) + ' : ' + '{}'.format(y).rjust(2))

    # Percentage of the data which should be used for training
    percentage = 0.7
    n = len(Y)

    # Split data into train and validation
    X_train, X_val, Y_train, Y_val = com.split_data(X, Y, percentage)

    # Number of iterations
    n_iter = 60

    optimise(n_iter, X, X_train, X_val, Y, Y_train, Y_val)

if __name__ == '__main__':
    main()
