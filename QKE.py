from distutils.util import execute
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
from qiskit.circuit.library import *
from qiskit import Aer, execute
from qiskit.tools.visualization import circuit_drawer
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
from QKE_functions import *
from classicalSVM import *
from data import *

def QKE(sample_train, sample_test, label_train, label_test, cross_fold, feature_dimension):
    zz_map = ZZFeatureMap(feature_dimension, reps = 1, entanglement="linear", insert_barriers=True)
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

    if cross_fold<=1:
        #Calculates accuracy without cross validation
        zzpc_svc.fit(matrix_train, label_train)
        zzpc_score = zzpc_svc.score(matrix_test, label_test)

        print("QKE accuracy: %0.3f\n" %(zzpc_score))
    else:
        #Calculates accuracy with cross validation, and presents mean and standard deviation
        scores=cross_val_score(zzpc_svc,matrix_train,label_train, cv=cross_fold)
        print("QKE accuracy: %0.3f ± %0.3f, Cross_fold ammount: %0.3f\n" % (scores.mean(), scores.std(), cross_fold))


def main():
   
    n_classes = 2
    n_attributes = 4
    n_data = 1000
    adhoc_dimension = 2

    #[sample_train, sample_test, label_train, label_test] = load_data_adhoc(50, adhoc_dimension)
    [sample_train, sample_test, label_train, label_test] = load_data_breast(n_attributes)

    cross_fold_QKE=5

    QKE(sample_train, sample_test, label_train, label_test, cross_fold_QKE, n_attributes)

    kernel_function=['linear', 'poly', 'rbf', 'sigmoid']
    poly_degree=2
    cross_fold_classical=5
    run_SVM(
        kernel_function,
        poly_degree,
        [sample_train, sample_test, label_train, label_test],
        cross_fold_classical
    )

if __name__ == '__main__':
    main()