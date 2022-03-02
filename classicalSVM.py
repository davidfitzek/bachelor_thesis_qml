from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("Accuracy:",accuracy_score(y_test, y_pred))


### Breast Cancer

breast_cancer = datasets.load_breast_cancer()

bc = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
bc
print(breast_cancer.feature_names)
print(breast_cancer.target_names)

X = breast_cancer.data
Y = breast_cancer.target

#print(X)
#print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

clf = SVC(kernel='linear',gamma='scale')
#clf = SVC(kernel="poly",degree=2,gamma='scale')

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print("Accuracy:",accuracy_score(y_test, y_pred))