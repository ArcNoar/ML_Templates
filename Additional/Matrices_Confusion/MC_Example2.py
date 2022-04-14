from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)

clf = LogisticRegression().fit(X_train, y_train)
y_predict = clf.predict(X_test)
cMatrix = confusion_matrix(y_test, y_predict)
print(cMatrix)