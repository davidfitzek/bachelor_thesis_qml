from distutils.util import execute
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
from qiskit.circuit.library import *
from qiskit import Aer, execute
from qiskit.tools.visualization import plot_histogram
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
from QKE_functions import *
from classicalSVM import *
from data import *

def QKE(sample_train, sample_test, label_train, label_test, cross_fold, feature_dimension, map_type, reps):

    if map_type == 'zz':
        map = ZZFeatureMap(feature_dimension, reps, entanglement="linear", insert_barriers=True)
    else:
        map = ZFeatureMap(feature_dimension, reps, insert_barriers=True)

    kernel = QuantumKernel(feature_map=map, quantum_instance=Aer.get_backend('statevector_simulator'))
    
    matrix_train = kernel.evaluate(x_vec=sample_train)
    matrix_test = kernel.evaluate(x_vec=sample_test, y_vec=sample_train)

    zzpc_svc = SVC(kernel='precomputed') #Uses the precomputed kernel and calculates the SVM
    

    #Plot the probabilites for the quantum states, circuit and kernel matrix 
    plot_probabilities(sample_train, kernel)
    plot_kernel(matrix_train, matrix_test)
    plot_curcuit(sample_train, kernel)
    #plt.show()
    if cross_fold<=1:
        #Calculates accuracy without cross validation
        zzpc_svc.fit(matrix_train, label_train)
        zzpc_score = zzpc_svc.score(matrix_test, label_test)
        print("QKE accuracy: %0.3f\n" %(zzpc_score))
    else:
        #Calculates accuracy with cross validation, and presents mean and standard deviation
        scores=cross_val_score(zzpc_svc,matrix_train,label_train, cv=cross_fold)
        print("QKE accuracy: %0.3f ± %0.3f, Cross_fold ammount: %0.1f\n" % (scores.mean(), scores.std(), cross_fold))


def plot_probabilities(sample_train, kernel):
    circuit = kernel.construct_circuit(sample_train[0], sample_train[1])
    job = execute(circuit, Aer.get_backend('qasm_simulator'), shots = 8192, seed_simulator=1024, seed_transpiler=1024)
    counts = job.result().get_counts(circuit)
    plot_histogram(counts)
    

def plot_kernel(matrix_train, matrix_test):
    fig, axs = plt.subplots(1,2,figsize=(10, 5))
    axs[0].imshow(np.asmatrix(matrix_train), 
                    interpolation='nearest', origin='upper', cmap='Blues')
    axs[0].set_title("Training kernel matrix")
    axs[1].imshow(np.asmatrix(matrix_test), 
                    interpolation='nearest', origin='upper', cmap='Reds')
    axs[1].set_title("Testing kernel matrix")

def plot_curcuit(sample_train, kernel):
    circuit = kernel.construct_circuit(sample_train[0], sample_train[1])
    circuit.decompose().decompose().draw(output='mpl')

def main():
    n_attributes = 4
    n_data = 150
    #'z' is a z feature map 
    #'zz' is a higher order zz feature map 
    map_type = 'z'
    
    reps = 2 #Repitition of layers in feature map

    #Loading data from data.py file
    [sample_train, sample_test, label_train, label_test] = load_data_forest(n_data, n_attributes)
    cross_fold_QKE=5

    QKE(sample_train, sample_test, label_train, label_test, cross_fold_QKE, n_attributes, map_type, reps)

    kernel_function=['linear', 'poly', 'rbf', 'sigmoid']
    poly_degree=2
    #Amount of parts the data is divided into for cross validation
    #The runtime will be increased by a factor of this number roughly
    #if crossfold<=1 no cross validation is done
    cross_fold_classical=5
    run_SVM(
        kernel_function,
        poly_degree,
        [sample_train, sample_test, label_train, label_test],
        cross_fold_classical
    )

if __name__ == '__main__':
    main()