from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from data import *
import warnings
warnings.filterwarnings('ignore')

def rbf_measure(i, k, cross_fold, data):
    [sample_train, sample_test, label_train, label_test] = data
    mean_scores = []
    stds = []
    parameters = []
    for g in i["gamma"]:
        for c in i["C"]:
            clf = SVC(kernel=k,gamma=g, C=c,random_state=0)
            if cross_fold<=1:
                #Calculates accuracy without cross validation
                clf.fit(sample_train, label_train)
                clf.fit(sample_train, label_train)
                score = clf.score(sample_test, label_test)
                #print("%s accuracy: %0.3f, Cross_fold ammount: %0.1f" % (k, score, cross_fold))
            else:
                #Calculates accuracy with cross validation, and presents mean and standard deviation
                score=cross_val_score(clf,sample_test,label_test, cv=cross_fold)
                #print("%s accuracy: %0.3f ± %0.3f, Cross_fold ammount: %0.1f" % (k, scores.mean(), scores.std(), cross_fold))
                mean_scores.append(score.mean())
                stds.append(score.std())
                parameters.append([g, c])
    return mean_scores, stds, parameters

def measure(i, k, cross_fold, data):
    [sample_train, sample_test, label_train, label_test] = data
    mean_scores = []
    stds = []
    parameters = []

    if k =="linear":
        parameter = "C"
    elif k =="poly":
        parameter = "degree"
    else:
        parameter = "coef0"

    for c in i[parameter]:
        if k =="linear":
            clf = SVC(kernel=k,random_state=0)
        elif k =="poly":
            clf = SVC(kernel=k,degree=c,random_state=0)
        else:
            clf = SVC(kernel=k,coef0=c,random_state=0)

        if cross_fold<=1:
            #Calculates accuracy without cross validation
            clf.fit(sample_train, label_train)
            clf.fit(sample_train, label_train)
            score = clf.score(sample_test, label_test)
            #print("%s accuracy: %0.3f, Cross_fold ammount: %0.1f" % (k, score, cross_fold))
        else:
            #Calculates accuracy with cross validation, and presents mean and standard deviation
            score=cross_val_score(clf,sample_test,label_test, cv=cross_fold)
            #print("%s accuracy: %0.3f ± %0.3f, Cross_fold ammount: %0.1f" % (k, scores.mean(), scores.std(), cross_fold))
            mean_scores.append(score.mean())
            stds.append(score.std())
            parameters.append(c)
    return mean_scores, stds, parameters

def results(k, mean_scores, stds, parameters):
    x = 0
    for i in range(len(mean_scores)):
        if mean_scores[i] == max(mean_scores):
            x = i
    print("Results for " + k)
    print("Best parameters are: %a" %(parameters[x]))
    print()
    print("%0.3f ± %0.3f" % (max(mean_scores), stds[x]))
    print("_____")

def run_SVM(data,cross_fold):
    # Set the parameters by cross-validation
    tuned_parameters = [
    {"kernel": "rbf", "gamma": [1e-3, 1e-4, 'scale'], "C": [1, 5, 10, 50, 100, 200, 500, 1000, 200000]},
    {"kernel": "linear", "C": [1, 2, 3, 4, 10, 20]},
    {"kernel": "poly", "degree": [2, 3, 4, 5, 6, 7]},
    {"kernel": "sigmoid", "coef0": [ -100000, -10000, -1000, -100, -10,-5, -2, 2, 5, 10, 100, 1000, 10000, 100000]},
    ]

    for i in tuned_parameters:
        k = i["kernel"]
        if k == "rbf":
            [mean_scores, stds, parameters] = rbf_measure(i, k, cross_fold, data)
            results(k, mean_scores, stds, parameters)
        elif k == "linear":
            [mean_scores, stds, parameters] = measure(i, k, cross_fold, data)
            results(k, mean_scores, stds, parameters)
        elif k == "poly":
            [mean_scores, stds, parameters] = measure(i, k, cross_fold, data)
            results(k, mean_scores, stds, parameters)
        else:
            [mean_scores, stds, parameters] = measure(i, k, cross_fold, data)
            results(k, mean_scores, stds, parameters)
                
        
def main():

    #data=load_data_forest_oscar(500)
    data=load_data_breast(30, 500)
     
    cross_fold=10

    #Run the SVM once normally
    run_SVM(
        data,
        cross_fold
    )
    
if __name__=='__main__':
    main()