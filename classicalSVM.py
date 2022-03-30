from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


"""
import pandas as pd

df = pd.read_csv('/Users/rohith/Documents/Datasets/Iris_dataset/iris.csv')
df = df.drop(['Id'],axis=1)
target = df['Species']
s = set()
for val in target:
    s.add(val)
s = list(s)
rows = list(range(100,150))
df = df.drop(df.index[rows])
"""


"""
### Iris

iris = datasets.load_iris()
print(iris.feature_names)
print(iris.target_names)

df = pd.DataFrame(iris.data, columns = iris.feature_names)

# Let us plot two features and two labels:
x = df['sepal length (cm)']
y = df['petal length (cm)']

setosa_x = x[:50]
setosa_y = y[:50]

versicolor_x = x[50:100]
versicolor_y = y[50:100]

# Plot it
plt.figure(figsize=(8,6))
plt.scatter(setosa_x,setosa_y,marker='+',color='green')
plt.scatter(versicolor_x,versicolor_y,marker='_',color='red')
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.show()

X = iris.data#[:100]
Y = iris.target#[:100]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

clf = SVC(kernel='linear',gamma='scale')

#clf.fit(x_train,y_train)

#for i in range(x_train.shape[0]): ## Looping through batches
#        X_batch, Y_batch = x_train[i], y_train[i]
#        clf.partial_fit(X_batch, Y_batch) ## Partially fitting data in batches

#y_pred = clf.predict(x_test)
#print("Accuracy:",accuracy_score(y_test, y_pred))

scores=cross_val_score(clf,X,Y, cv=3)
print("Accuracy: %0.2f ± %0.2f" % (scores.mean(), scores.std()))



### Breast Cancer

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
#clf.fit(x_train,y_train)
#y_pred = clf.predict(x_test)
#print("Accuracy:",accuracy_score(y_test, y_pred))
"""


def run_SVM(kernel_function,poly_degree,data,cross_fold):
    X=data.data
    Y=data.target

    clf = SVC(kernel=kernel_function,degree=poly_degree,gamma='scale')

    #Calculates accuracy with cross validation, and presents mean and standard deviation
    scores=cross_val_score(clf,X,Y, cv=cross_fold)
    print("Accuracy: %0.2f ± %0.2f" % (scores.mean(), scores.std()))

def load_data_iris():
    iris = datasets.load_iris()

    #X = iris.data#[:100]
    #Y = iris.target#[:100]

    return iris

def load_data_cancer():
    breast_cancer = datasets.load_breast_cancer()

    bc = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
    bc
    print(breast_cancer.feature_names)
    print(breast_cancer.target_names)

    #X = breast_cancer.data
    #Y = breast_cancer.target
    return breast_cancer

def main():
    #Kernel choices:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    kernel_function='linear'
    #poly_degree is only relevant if kernel='poly' otherwise ignored
    poly_degree=2

    data=load_data_iris()

    #Ammount of parts the data is divided into (The runtime will be increased by a factor of this number roughly)
    cross_fold=3

    run_SVM(
        kernel_function,
        poly_degree,
        data,
        cross_fold
    )


if __name__=='__main__':
    main()