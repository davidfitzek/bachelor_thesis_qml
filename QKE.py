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
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, BasicAer, execute
from qiskit.tools.visualization import circuit_drawer
from urllib3 import encode_multipart_formdata
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
from QKE_functions import *

# Data import and manipulation

# Forest dataset
forest = datasets.fetch_covtype()

X_raw = forest.data
Y_raw = forest.target

X = []
Y = []

cnt = 0
for x, y in zip(X_raw, Y_raw):
    if y < 3:
        X.append(np.array(x))
        Y.append(y)

        cnt = cnt + 1
        if cnt == 500:
            break

forest_sample_train, forest_sample_test, forest_label_train, forest_label_test = train_test_split(
    X, Y, test_size=0.3, random_state=22)
[forest_sample_train, forest_sample_test] = reduceClassDimensions(4, forest_sample_train, forest_sample_test)

# Iris dataset
iris = datasets.load_iris()
iris_sample_train, iris_sample_test, iris_label_train, iris_label_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=22)

# Breastcancer dataset
breast = datasets.load_breast_cancer()
breast_sample_train, breast_sample_test, breast_label_train, breast_label_test = train_test_split(
    breast.data, breast.target, test_size=0.3, random_state=22)
[breast_sample_train, breast_sample_test] = reduceClassDimensions(4, breast_sample_train, breast_sample_test)

# Digits dataset
digits = datasets.load_digits(n_class=2)
digits_sample_train, digits_sample_test, digits_label_train, digits_label_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=22)
[digits_sample_train, digits_sample_test] = reduceClassDimensions(4, digits_sample_train, digits_sample_test)

# Choose your set to use the QKE
sample_train = forest_sample_train
label_train = forest_label_train

sample_test = forest_sample_test
label_test = forest_label_test


zz_map = ZZFeatureMap(feature_dimension=4, reps = 2, entanglement="linear", insert_barriers=True)
zz_kernel = QuantumKernel(feature_map=zz_map, quantum_instance=Aer.get_backend('statevector_simulator'))
zz_circuit = zz_kernel.construct_circuit(sample_train[0], sample_train[1])
zz_circuit.decompose().decompose().draw(output='mpl')

backend = Aer.get_backend('qasm_simulator')
job = execute(zz_circuit, backend, shots = 8192, seed_simulator=1024, seed_transpiler=1024)
counts = job.result().get_counts(zz_circuit)

# Compute and plot the kernel matrix
matrix_train = zz_kernel.evaluate(x_vec=sample_train)
matrix_test = zz_kernel.evaluate(x_vec=sample_test, y_vec=sample_train)

fig, axs = plt.subplots(1,2,figsize=(10, 5))
axs[0].imshow(np.asmatrix(matrix_train), 
                interpolation='nearest', origin='upper', cmap='Blues')
axs[0].set_title("Training kernel matrix")
axs[1].imshow(np.asmatrix(matrix_test), 
                interpolation='nearest', origin='upper', cmap='Reds')
axs[1].set_title("Testing kernel matrix")
#plt.show()

zzpc_svc = SVC(kernel='precomputed')
zzpc_svc.fit(matrix_train, label_train)

#for i in range(matrix_train.shape[0]): ## Looping through batches
#        X_batch, Y_batch = matrix_train[i], label_train[i]
#        zzpc_svc.partial_fit(X_batch, Y_batch) ## Partially fitting data in batches

zzpc_score = zzpc_svc.score(matrix_test, label_test)

print(f'Quantum kernel classification score: {zzpc_score}\n')

# Classical QSVM comparison

classical_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for k in classical_kernels:
    classical_svc = SVC(kernel=k)
    classical_svc.fit(sample_train, label_train)
    classical_score = classical_svc.score(sample_test, label_test)
    print('%s kernel classification score: %0.5f' % (k, classical_score))