from configparser import Interpolation
from ctypes import sizeof
from distutils.util import execute
from hmac import trans_36
from unicodedata import digit
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
from qiskit.circuit.library import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, BasicAer, execute
from qiskit.tools.visualization import circuit_drawer
from urllib3 import encode_multipart_formdata
from qiskit_machine_learning.kernels import QuantumKernel
from QKE_functions import *
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC

seed = 12345
algorithm_globals.random_seed = seed

adhoc_dimension = 2
train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    training_size=20,
    test_size=5,
    n=adhoc_dimension,
    gap=0.3,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)

adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")

adhoc_backend = QuantumInstance(
    BasicAer.get_backend("qasm_simulator"), shots=1024, seed_simulator=seed, seed_transpiler=seed
)

adhoc_kernel = QuantumKernel(feature_map=adhoc_feature_map, quantum_instance=adhoc_backend)

adhoc_svc = SVC(kernel=adhoc_kernel.evaluate)
adhoc_svc.fit(train_features, train_labels)
adhoc_score = adhoc_svc.score(test_features, test_labels)

print(f"Callable kernel classification test score: {adhoc_score}")

adhoc_matrix_train = adhoc_kernel.evaluate(x_vec=train_features)
adhoc_matrix_test = adhoc_kernel.evaluate(x_vec=test_features, y_vec=train_features)

adhoc_svc = SVC(kernel="precomputed")
adhoc_svc.fit(adhoc_matrix_train, train_labels)
adhoc_score = adhoc_svc.score(adhoc_matrix_test, test_labels)

print(f"Precomputed kernel classification test score: {adhoc_score}")

qsvc = QSVC(quantum_kernel=adhoc_kernel)
qsvc.fit(train_features, train_labels)
qsvc_score = qsvc.score(test_features, test_labels)

print(f"QSVC classification test score: {qsvc_score}")

classical_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for k in classical_kernels:
    classical_svc = SVC(kernel=k)
    classical_svc.fit(train_features, train_labels)
    classical_score = classical_svc.score(test_features, test_labels)
    print('%s kernel classification score: %0.5f' % (k, classical_score))