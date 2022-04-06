import matplotlib.pyplot as plt
from numpy import not_equal
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AngleEmbedding, BasisEmbedding, AmplitudeEmbedding
from data import *
from QKE_functions import *
from classicalSVM import *

# amplitude, n attributes = 8, n_data = 150: 0.933, cross = 2 
# angle, n attributes = 4, n_data = 150: 0.952, cross = 5

#Choose kernelfunction: [kernel_angle, kernel_basis, kernel_amplitude]
kernel_name = 'kernel_angle'
#Amount of features used for the dataset
n_features = 4
#If amplitude encoding, n_qubits = log_2(n_features)
n_qubits = np.int64(np.ceil(np.log2(n_features))) if kernel_name == 'kernel_amplitude' else n_features
n_wires = n_qubits
#Create the zero projector
projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

dev = qml.device("default.qubit", wires = n_wires)
@qml.qnode(dev)
def kernel_angle(x, y):
    """Kernel function with angle encoding. This circuit will rotate the 
        N-dimensional input data into N qubits. Input data can be floatnumbers."""
    AngleEmbedding(x, wires=range(n_wires))
    qml.adjoint(AngleEmbedding)(y, wires=range(n_wires))
    return qml.expval(qml.Hermitian(projector, wires=range(n_wires)))

@qml.qnode(dev)
def kernel_basis(x, y):
    """Kernel function with basis encoding. This circuit will encode N 
        input data into N qubits. Input data can only be 0 or 1"""
    BasisEmbedding(x, wires=range(n_wires))
    qml.adjoint(BasisEmbedding)(y, wires=range(n_wires))
    return qml.expval(qml.Hermitian(projector, wires=range(n_wires)))

@qml.qnode(dev)
def kernel_amplitude(x, y):
    """Kernel function with amplitude encoding. This circuit will encode the 
        N-dimensional input data into the amplitudes of log(N) qubits.
        Input data can be floatnumbers."""
    AmplitudeEmbedding(x, wires=range(n_wires), normalize=True)
    qml.adjoint(AmplitudeEmbedding)(y, wires=range(n_wires), normalize=True)
    return qml.expval(qml.Hermitian(projector, wires=range(n_wires)))


def kernel_matrix(A, B, kernel_function):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel_function(a, b) for b in B] for a in A])

def main():

    #Load the data
    #load_data_adhoc(150, 2)
    #load_data_breast(n_attributes=n_features, n_data = 150)
    #load_data_iris(150)
    #load_data

    [sample_train, sample_test, label_train, label_test] = load_data_breast(n_attributes=4, n_data = 150)

    #Scale the data
    [sample_train, sample_test] = scale(sample_train, sample_test, -1, 1)

    if kernel_name == 'kernel_angle':
        kernel_function =  kernel_angle
    elif kernel_name == 'kernel_basis':
        kernel_function =  kernel_basis
    else:
        kernel_function =  kernel_amplitude

    #Amount of parts the data is divided into for cross validation
    #The runtime will be increased by a factor of this number roughly
    #if crossfold<=1 no cross validation is done
    cross_fold = 5

    #Calculate the kernel matrices
    matrix_train = kernel_matrix(sample_train, sample_train, kernel_function)
    matrix_test = kernel_matrix(sample_test, sample_train, kernel_function)

    #Calculate the SVM classically with the Quantum Kernel
    qsvm = SVC(kernel='precomputed')

    if cross_fold<=1:
        #Calculates accuracy without cross validation
        qsvm.fit(matrix_train, label_train)
        score = qsvm.score(matrix_test, label_test)
        print("QKE accuracy: %0.3f\n" %(score))
    else:
        #Calculates accuracy with cross validation, and presents mean and standard deviation
        scores_cross=cross_val_score(qsvm,matrix_train,label_train, cv=cross_fold)
        print("QKE accuracy: %0.3f ± %0.3f, Cross_fold ammount: %0.1f\n" % (scores_cross.mean(), scores_cross.std(), cross_fold))


    #Print the classical results
    kernel_function=['linear', 'poly', 'rbf', 'sigmoid']
    poly_degree=2
    #Cross_fold for classical SVM
    cross_fold_classical=5
    run_SVM(
        kernel_function,
        poly_degree,
        [sample_train, sample_test, label_train, label_test],
        cross_fold_classical
    )

if __name__ == '__main__':
    main()