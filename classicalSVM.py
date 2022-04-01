from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_SVM(kernel_function,poly_degree,data,cross_fold):
    X=data.data
    Y=data.target
    for k in kernel_function:
        clf = SVC(kernel=k,degree=poly_degree,gamma='scale')

        if cross_fold<=1:
            #Calculates accuracy without cross validation
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
            clf.fit(x_train,y_train)
            y_pred = clf.predict(x_test)
            if k == 'poly':
                print("%s accuracy: %0.2f, Cross_fold ammount: %0.2f, Poly_degree: %0.2f" % (k, accuracy_score(y_test, y_pred), cross_fold, poly_degree))
            else:
                print("%s accuracy: %0.2f, Cross_fold ammount: %0.2f" % (k, accuracy_score(y_test, y_pred), cross_fold))
        else:
            #Calculates accuracy with cross validation, and presents mean and standard deviation
            scores=cross_val_score(clf,X,Y, cv=cross_fold)
            if k == 'poly':
                print("%s accuracy: %0.2f ± %0.2f, Cross_fold ammount: %0.2f, Poly_degree: %0.2f" % (k, scores.mean(), scores.std(), cross_fold, poly_degree))
            else:
                print("%s accuracy: %0.2f ± %0.2f, Cross_fold ammount: %0.2f" % (k, scores.mean(), scores.std(), cross_fold))

def load_data_iris():
    iris = datasets.load_iris()

    return iris

def load_data_cancer():
    breast_cancer = datasets.load_breast_cancer()
    return breast_cancer

def main():
    #Kernel choices:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    kernel_function=['linear', 'poly', 'rbf', 'sigmoid']
    #poly_degree is only relevant if kernel='poly' otherwise ignored
    poly_degree=2

    data=load_data_iris()

    #Amount of parts the data is divided into for cross validation
    #The runtime will be increased by a factor of this number roughly
    #if crossfold<=1 no cross validation is done
    cross_fold=5

    #Run the SVM once normally
    run_SVM(
        kernel_function,
        poly_degree,
        data,
        cross_fold
    )
    
if __name__=='__main__':
    main()