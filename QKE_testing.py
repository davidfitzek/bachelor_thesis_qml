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
from QKE_functions import *

forest = datasets.fetch_covtype()
forest_sample_train, forest_sample_test, forest_label_train, forest_label_test = train_test_split(
    forest.data, forest.target, test_size=0.3, random_state=14)

[forest_sample_train, forest_sample_test] = reduceClassDimensions(4, forest_sample_train, forest_sample_test)

print(forest_sample_train, forest_label_train)