from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from data import *

def run_SVM(kernel_function,poly_degree,data,cross_fold):
    [sample_train, sample_test, label_train, label_test] = data
    for k in kernel_function:
        clf = SVC(kernel=k,degree=poly_degree, random_state=0, C = 10)
        if cross_fold<=1:
            #Calculates accuracy without cross validation
            clf.fit(sample_train, label_train)
            clf.fit(sample_train, label_train)
            score = clf.score(sample_test, label_test)
            if k == 'poly':
                print("%s accuracy: %0.3f, Cross_fold ammount: %0.1f, Poly_degree: %0.1f" % (k, score, cross_fold, poly_degree))
            else:
                print("%s accuracy: %0.3f, Cross_fold ammount: %0.1f" % (k, score, cross_fold))
        else:
            #Calculates accuracy with cross validation, and presents mean and standard deviation
            scores=cross_val_score(clf,sample_test,label_test, cv=cross_fold)
            if k == 'poly':
                print("%s accuracy: %0.3f ± %0.3f, Cross_fold ammount: %0.1f, Poly_degree: %0.1f" % (k, scores.mean(), scores.std(), cross_fold, poly_degree))
            else:
                print("%s accuracy: %0.3f ± %0.3f, Cross_fold ammount: %0.1f" % (k, scores.mean(), scores.std(), cross_fold))

def main():
    kernel_function=['linear', 'poly', 'rbf', 'sigmoid']

    poly_degree=2

    #Load the data
    [sample_train, sample_test, label_train, label_test]=load_data_iris(100)
    [sample_train, sample_test] = scale(sample_train, sample_test, -1, 1)
    [sample_train, sample_test] = normalise(sample_train, sample_test)
    cross_fold=10

    data = [sample_train, sample_test, label_train, label_test]
    #Run the SVM once normally
    run_SVM(
        kernel_function,
        poly_degree,
        data,
        cross_fold
    )
    
if __name__=='__main__':
    main()