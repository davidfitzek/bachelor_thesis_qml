from configparser import Interpolation
from ctypes import sizeof
from distutils.util import execute
from hmac import trans_36
from sys import orig_argv
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
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute
from qiskit.tools.visualization import circuit_drawer
from urllib3 import encode_multipart_formdata
from qiskit_machine_learning.kernels import QuantumKernel


# Data import and manipulation

# Data manipulation for iris dataset
iris = datasets.load_iris()

# Split the dataset
iris_sample_train, iris_sample_test, iris_label_train, iris_label_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=10)


# Data manipulation for the digits dataset
digits = datasets.load_digits(n_class=2)

# Split the dataset
digits_sample_train, digits_sample_test, digits_label_train, digits_label_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=22)

# Reduce dimensions
n_dim = 4
pca = PCA(n_components=n_dim).fit(digits_sample_train)
digits_sample_train = pca.transform(digits_sample_train)
digits_sample_test = pca.transform(digits_sample_test)

# Normalise
std_scale = StandardScaler().fit(digits_sample_train)
digits_sample_train = std_scale.transform(digits_sample_train)
digits_sample_test = std_scale.transform(digits_sample_test)

# Scale
digits_samples = np.append(digits_sample_train, digits_sample_test, axis=0)
minmax_scale = MinMaxScaler((-1, 1)).fit(digits_samples)
digits_sample_train = minmax_scale.transform(digits_sample_train)
digits_sample_test = minmax_scale.transform(digits_sample_test)


sample_train = iris_sample_train#digits_sample_train[:100]
label_train = iris_label_train#digits_label_train[:100]

sample_test = iris_sample_test#digits_sample_test[:20]
label_test = iris_label_test#digits_label_test[:20]


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
zzpc_score = zzpc_svc.score(matrix_test, label_test)

print(f'Quantum kernel classification score: {zzpc_score}\n')

# Classical QSVM comparison

classical_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for k in classical_kernels:
    classical_svc = SVC(kernel=k)
    classical_svc.fit(sample_train, label_train)
    classical_score = classical_svc.score(sample_test, label_test)
    print('%s kernel classification score: %0.5f' % (k, classical_score))
