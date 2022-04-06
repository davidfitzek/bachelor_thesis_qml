import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AngleEmbedding, BasisEmbedding, AmplitudeEmbedding
from data import *
from QKE_functions import *

n_features = 4
n_qubits = np.int64(np.ceil(np.log2(n_features)))
n_qubits = 4
n_wires = n_qubits

dev = qml.device("default.qubit", wires = n_wires)

projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

@qml.qnode(dev)
def kernel_angle(x, y):
    wires = n_qubits
    AngleEmbedding(x, wires=range(wires))
    qml.adjoint(AngleEmbedding)(y, wires=range(wires))
    return qml.expval(qml.Hermitian(projector, wires=range(wires)))

@qml.qnode(dev)
def kernel_basis(x, y):
    wires = n_qubits
    BasisEmbedding(x, wires=range(wires))
    qml.adjoint(BasisEmbedding)(y, wires=range(wires))
    return qml.expval(qml.Hermitian(projector, wires=range(wires)))

@qml.qnode(dev)
def kernel_amplitude(x, y):
    wires = n_qubits
    AmplitudeEmbedding(x, wires=range(wires), normalize=True)
    qml.adjoint(AmplitudeEmbedding)(y, wires=range(wires), normalize=True)
    return qml.expval(qml.Hermitian(projector, wires=range(wires)))


def kernel_matrix(A, B, kernel_function):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel_function(a, b) for b in B] for a in A])

def main():

    n_data = 100

    cross_fold = 2

    kernel_function = kernel_angle

    [sample_train, sample_test, label_train, label_test] = load_data_forest(n_data, n_attributes = 4)
    
    #print(sample_train, label_train)

    #[sample_train, sample_test] = normalise(sample_train, sample_test)
    [sample_train, sample_test] = scale(sample_train, sample_test, -1, 1)

    matrix_train = kernel_matrix(sample_train, sample_train, kernel_function)
    matrix_test = kernel_matrix(sample_test, sample_train, kernel_function)

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

if __name__ == '__main__':
    main()