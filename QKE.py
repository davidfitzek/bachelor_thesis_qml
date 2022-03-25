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
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.tools.visualization import circuit_drawer
from urllib3 import encode_multipart_formdata
from qiskit_machine_learning.kernels import QuantumKernel



digits = datasets.load_digits(n_class=2)
iris = datasets.load_iris()


# Data manipulation for the digits dataset

# Split the dataset
sample_train, sample_test, label_train, label_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=22)

# Reduce dimensions
n_dim = 4
pca = PCA(n_components=n_dim).fit(sample_train)
sample_train = pca.transform(sample_train)
sample_test = pca.transform(sample_test)

# Normalise
std_scale = StandardScaler().fit(sample_train)
sample_train = std_scale.transform(sample_train)
sample_test = std_scale.transform(sample_test)

# Scale
samples = np.append(sample_train, sample_test, axis=0)
minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
sample_train = minmax_scale.transform(sample_train)
sample_test = minmax_scale.transform(sample_test)

zz_map = ZZFeatureMap(feature_dimension=4, reps = 2, entanglement="linear", insert_barriers=True)
zz_kernel = QuantumKernel()




