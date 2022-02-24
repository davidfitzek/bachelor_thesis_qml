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

