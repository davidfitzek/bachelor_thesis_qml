# iris_classifier.py

import statistics

import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as opt

import common as com
import data as dat

import time

import csv

import numpy

np.random.seed(123)  # Set seed for reproducibility


# The layer for the circuit
def layer_ex1(weights):
    n = len(weights)

    # Adds rotation matrices
    for i, row in enumerate(weights):
        qml.Rot(row[0], row[1], row[2], wires=i)

    # Adds controlled NOT matrices
    for i in range(n):
        qml.CNOT(wires=[i, (i + 1) % n])


def layer_ex2(weights):
    n = len(weights)

    # Adds rotation matrices and controlled NOT matrices
    for i, row in enumerate(weights):
        qml.Rot(row[0], row[1], row[2], wires=i)
        qml.CNOT(wires=[i, (i + 1) % n])


def stateprep_amplitude(features):
    wires = np.int64(np.ceil(np.log2(len(features))))
    # Normalise the features here and also pad it to have the length of a power of two
    qml.AmplitudeEmbedding(features=features, wires=range(wires), pad_with=0, normalize=True)


def stateprep_angle(features):
    wires = len(features)
    qml.AngleEmbedding(features=features, wires=range(wires), rotation='Y')


# The circuit function, allows variable statepreparation and layer functions
def circuit_fun(weights, features, stateprep_fun, layer_fun):
    stateprep_fun(features)

    for weight in weights:
        layer_fun(weight)

    return qml.expval(qml.PauliZ(0))


def variational_classifier_fun(weights, features, bias, circuit_fun):
    return circuit_fun(weights, features) + bias


def cost_fun(weights, bias, features, labels, variational_classifier_fun):
    preds = [variational_classifier_fun(weights, feature, bias) for feature in features]
    return com.square_loss(labels, preds)


def optimise(accuracy_stop, cost_stop, iter_stop, weights, bias, data, data_train, data_val, circuit, n_layers,
             cross_iter, cross_res, iter_res, cost_res, accu_res):
    optimiser = opt.NesterovMomentumOptimizer(stepsize=0.01)  # Performs much better than GradientDescentOptimizer
    # optimiser = opt.AdamOptimizer(stepsize = 0.01) # To be tried, was mentioned
    # optimiser = opt.GradientDescentOptimizer(stepsize = 0.01)
    batch_size = 5  # This might be something which can be adjusted

    # Variational classifier function used by pennylane
    def variational_classifier(weights, features, bias):
        return variational_classifier_fun(weights, features, bias, circuit)

    # Cost function used by pennylane
    def cost(weights, bias, features, labels):
        return cost_fun(weights, bias, features, labels, variational_classifier)

    # Number of training points, used when choosing batch indexes
    n_train = data_train.size()

    accuracy_val = 0.0
    cost_var = 100  # just something big
    i = 0
    while i < iter_stop and ((accuracy_val < accuracy_stop) or (cost_var > cost_stop)):
        # Update the weights by one optimiser step
        batch_index = np.random.randint(0, high=n_train, size=(batch_size,))
        X_train_batch = data_train.X[batch_index]
        Y_train_batch = data_train.Y[batch_index]
        weights, bias, _, _ = optimiser.step(cost, weights, bias, X_train_batch, Y_train_batch)
        # Compute predictions on train and test set
        predictions_train = [np.sign(variational_classifier(weights, x, bias)) for x in data_train.X]
        predictions_val = [np.sign(variational_classifier(weights, x, bias)) for x in data_val.X]

        # Compute accuracy on train and test set
        accuracy_train = com.accuracy(data_train.Y, predictions_train)
        accuracy_val = com.accuracy(data_val.Y, predictions_val)

        cost_var = float(cost(weights, bias, data.X, data.Y))

        print(
            'Cross validation iteration: {:d} | Iteration: {:d} | Cost: {:0.7f} | Accuracy train: {:0.7f} | Accuracy validation: {:0.7f} | Layers: {:d} '
            ''.format(cross_iter + 1, i + 1, cost_var, accuracy_train, accuracy_val, n_layers))

        cross_res.append(cross_iter + 1)
        iter_res.append(i + 1)
        cost_res.append(cost_var)
        accu_res.append(float(accuracy_val))

        i += 1

    return [i, cost_var, float(accuracy_val), cross_res, iter_res, cost_res, accu_res]


# Shuffles the data points
def shuffle_data(data):
    N = data.size()  # Number of data points

    indexes = np.random.permutation(N)

    return dat.Data(data.X[indexes], data.Y[indexes])


def run_variational_classifier(cross_fold, n_qubits, n_layers, data, stateprep_fun, layer_fun, accuracy_stop, cost_stop,
                               iter_stop):
    # The device and qnode used by pennylane
    device = qml.device("default.qubit", wires=n_qubits)

    # Circuit function used by pennylane
    @qml.qnode(device)
    def circuit(weights, x):
        return circuit_fun(weights, x, stateprep_fun, layer_fun)

    # Shuffle our data to introduce a random element to our train and test parts
    data = shuffle_data(data)

    # Compute the size of
    N = data.size()
    cross_size = N // cross_fold

    iteration_best = []
    cost_best = []
    accuracy_best = []
    cross_res = []
    iter_res = []
    cost_res = []
    accu_res = []

    for cross_iter in range(cross_fold):
        data_train, data_val = dat.split_data(data, cross_iter * cross_size, (cross_iter + 1) * cross_size)

        weights = 0.01 * np.random.randn(n_layers, n_qubits, 3, requires_grad=True)  # Initial value for the weights
        bias = np.array(0.0, requires_grad=True)  # Initial value for the bias
        tmp = optimise(accuracy_stop, cost_stop, iter_stop, weights, bias, data, data_train, data_val, circuit,
                       n_layers, cross_iter, cross_res, iter_res, cost_res, accu_res)
        iteration_best.append(tmp[0])
        cost_best.append(tmp[1])
        accuracy_best.append(tmp[2])
    # cross_res.append(tmp[3])
    # iter_res.append(tmp[4])
    # cost_res.append(tmp[5])
    # accu_res.append(tmp[6])

    return [iteration_best, cost_best, accuracy_best, cross_res, iter_res, cost_res, accu_res]


def main():
    # it will test all the number of layers up to this number
    range_layers = 10

    # if the accuracy validation is higher and the cost is lower or if the iterations are higher it stops
    accuracy_stop = 0.8
    cost_stop = 0.3
    iter_stop = 100

    cross_fold = 10  # The amount of parts the data is divided into, =1 gives no cross validation

    # Can be any function which takes in a matrix of weights and creates a layer
    layer_fun = layer_ex1

    # Can be any function that takes an input vector and encodes it
    stateprep_name = ["Angle"]  # descriptive name
    stateprep_array = [stateprep_angle]

    # Load data
    data_name = ["Adhoc3"]  # descriptive name
    data_array = [dat.load_data_adhoc(dimensions = 3)]

    start_time = time.perf_counter()

    # Tests every state preparation
    for which_stateprep in range(len(stateprep_array)):
        print("Now starting with state preparation " + stateprep_name[which_stateprep] + "\n")
        stateprep_fun = stateprep_array[which_stateprep]

        # Corresponds to the first stateprep encoding in stateprep_array
        if which_stateprep == 0:  # Kind of ugly way to do it but I do not know of a better one
            qubit_array = [3]

        # Tests every dataset
        for which_data in range(len(data_array)):
            print("Dataset: " + data_name[which_data] + "\n")
            n_qubits = qubit_array[which_data]
            data = data_array[which_data]

            # testing how many layers it takes to achieve accuracy_stop and cost_stop
            iterMean = [0] * range_layers
            iterDev = [0] * range_layers
            costMean = [0] * range_layers
            costDev = [0] * range_layers
            accuMean = [0] * range_layers
            accuDev = [0] * range_layers
            sec = [0] * range_layers

            for i in range(range_layers):
                n_layers = i + 1
                print("Layer " + str(n_layers) + " of " + str(range_layers) + "\n")
                tic = time.perf_counter()
                tmp = run_variational_classifier(cross_fold, n_qubits, n_layers, data, stateprep_fun, layer_fun,
                                                 accuracy_stop, cost_stop, iter_stop)
                toc = time.perf_counter()

                iteration_best = tmp[0]
                cost_best = tmp[1]
                accuracy_best = tmp[2]
                cross_res = tmp[3]
                iter_res = tmp[4]
                cost_res = tmp[5]
                accu_res = tmp[6]

                iterMean[i] = statistics.mean(iteration_best)
                iterDev[i] = statistics.stdev(iteration_best)
                costMean[i] = statistics.mean(cost_best)
                costDev[i] = statistics.stdev(cost_best)
                accuMean[i] = statistics.mean(accuracy_best)
                accuDev[i] = statistics.stdev(accuracy_best)
                sec[i] = toc - tic

                print(" ")
                print("It took " + str(sec[i]) + " seconds")
                print('Final Accuracy: {:0.7f} +- {:0.7f}'.format(statistics.mean(accuracy_best),
                                                                  statistics.stdev(accuracy_best)))
                print('Final Cost: {:0.7f} +- {:0.7f}'.format(statistics.mean(cost_best),
                                                              statistics.stdev(cost_best)) + "\n")

                # save the data as a file
                fields = ["Cross iteration", "Iteration", "Cost", "Accuracy"]
                res = []
                res.append(cross_res)
                res.append(iter_res)
                res.append(cost_res)
                res.append(accu_res)
                res = numpy.array(res).T.tolist()

                with open("./Adhoc/" + data_name[which_data] + stateprep_name[which_stateprep] + "Layer" + str(
                        n_layers) + ".csv", "w") as f:
                    write = csv.writer(f)

                    write.writerow(fields)
                    write.writerows(res)

            # Save the data as a file
            fields = ["Iterations mean", "Iterations dev", "Cost mean", "Cost dev", "Accuracy mean", "Accuracy dev",
                      "Seconds"]
            res = []
            res.append(iterMean)
            res.append(iterDev)
            res.append(costMean)
            res.append(costDev)
            res.append(accuMean)
            res.append(accuDev)
            res.append(sec)
            res = numpy.array(res).T.tolist()

            with open("./Adhoc/" + data_name[which_data] + stateprep_name[which_stateprep] + ".csv", "w") as f:
                write = csv.writer(f)

                write.writerow(fields)
                write.writerows(res)

    stop_time = time.perf_counter()
    total_time = (stop_time - start_time) / 3600.0

    print("Done! It took " + str(total_time) + " hours to run this programme.")


if __name__ == '__main__':
    main()
